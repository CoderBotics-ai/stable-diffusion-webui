from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Dict, Any, Type

import lyco_helpers
import modules.models.sd3.mmdit
import network
from modules import devices


class ModuleTypeLora(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[NetworkModuleLora]:
        """Create a LoRA network module based on the weights structure."""
        if all(x in weights.w for x in ["lora_up.weight", "lora_down.weight"]):
            return NetworkModuleLora(net, weights)

        if all(x in weights.w for x in ["lora_A.weight", "lora_B.weight"]):
            w = weights.w.copy()
            weights.w.clear()
            weights.w.update({
                "lora_up.weight": w["lora_B.weight"],
                "lora_down.weight": w["lora_A.weight"]
            })
            return NetworkModuleLora(net, weights)

        return None


class NetworkModuleLora(network.NetworkModule):
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        super().__init__(net, weights)

        self.up_model: nn.Module = self.create_module(weights.w, "lora_up.weight")
        self.down_model: nn.Module = self.create_module(weights.w, "lora_down.weight")
        self.mid_model: Optional[nn.Module] = self.create_module(weights.w, "lora_mid.weight", none_ok=True)

        self.dim: int = weights.w["lora_down.weight"].shape[0]

    def create_module(self, weights: Dict[str, torch.Tensor], key: str, none_ok: bool = False) -> Optional[nn.Module]:
        """Create a neural network module based on the weight configuration."""
        weight = weights.get(key)

        if weight is None and none_ok:
            return None

        supported_linear_types: tuple[Type, ...] = (
            nn.Linear,
            nn.modules.linear.NonDynamicallyQuantizableLinear,
            nn.MultiheadAttention,
            modules.models.sd3.mmdit.QkvLinear
        )
        
        is_linear = isinstance(self.sd_module, supported_linear_types)
        is_conv = isinstance(self.sd_module, nn.Conv2d)

        if is_linear:
            weight = weight.reshape(weight.shape[0], -1)
            module = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif is_conv and (key == "lora_down.weight" or key == "dyn_up"):
            if len(weight.shape) == 2:
                weight = weight.reshape(weight.shape[0], -1, 1, 1)

            if weight.shape[2] != 1 or weight.shape[3] != 1:
                module = nn.Conv2d(
                    weight.shape[1],
                    weight.shape[0],
                    self.sd_module.kernel_size,
                    self.sd_module.stride,
                    self.sd_module.padding,
                    bias=False
                )
            else:
                module = nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif is_conv and key == "lora_mid.weight":
            module = nn.Conv2d(
                weight.shape[1],
                weight.shape[0],
                self.sd_module.kernel_size,
                self.sd_module.stride,
                self.sd_module.padding,
                bias=False
            )
        elif is_conv and (key == "lora_up.weight" or key == "dyn_down"):
            module = nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            raise AssertionError(
                f'Lora layer {self.network_key} matched a layer with unsupported type: '
                f'{type(self.sd_module).__name__}'
            )

        with torch.no_grad():
            if weight.shape != module.weight.shape:
                weight = weight.reshape(module.weight.shape)
            module.weight.copy_(weight)

        module.to(device=devices.cpu, dtype=devices.dtype)
        module.weight.requires_grad_(False)

        return module

    def calc_updown(self, orig_weight: torch.Tensor) -> torch.Tensor:
        """Calculate the up-down composition of the LoRA weights."""
        up = self.up_model.weight.to(orig_weight.device)
        down = self.down_model.weight.to(orig_weight.device)

        output_shape = [up.size(0), down.size(1)]
        if self.mid_model is not None:
            # cp-decomposition
            mid = self.mid_model.weight.to(orig_weight.device)
            updown = lyco_helpers.rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = lyco_helpers.rebuild_conventional(up, down, output_shape, self.network.dyn_dim)

        return self.finalize_updown(updown, orig_weight, output_shape)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LoRA module."""
        self.up_model.to(device=devices.device)
        self.down_model.to(device=devices.device)

        return y + self.up_model(self.down_model(x)) * self.multiplier() * self.calc_scale()