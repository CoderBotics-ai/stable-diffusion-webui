from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Union
import enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import sd_models, cache, errors, hashes, shared
import modules.models.sd3.mmdit

@dataclass
class NetworkWeights:
    network_key: str
    sd_key: str
    w: dict[str, Any]
    sd_module: Union[nn.Module, Any]

# Using TypedDict for structured metadata
metadata_tags_order: dict[str, int] = {
    "ss_sd_model_name": 1,
    "ss_resolution": 2,
    "ss_clip_skip": 3,
    "ss_num_train_images": 10,
    "ss_tag_frequency": 20
}


class SdVersion(enum.Enum):
    """Enumeration of supported SD model versions."""
    Unknown = 1
    SD1 = 2
    SD2 = 3
    SDXL = 4


class NetworkOnDisk:
    def __init__(self, name: str, filename: str) -> None:
        self.name: str = name
        self.filename: str = filename
        self.metadata: dict[str, Any] = {}
        self.is_safetensors: bool = os.path.splitext(filename)[1].lower() == ".safetensors"
        self.alias: str = ""
        self.hash: Optional[str] = None
        self.shorthash: Optional[str] = None

        def read_metadata() -> dict[str, Any]:
            return sd_models.read_metadata_from_safetensors(filename)

        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "lora/" + self.name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading lora {filename}")

        if self.metadata:
            self.metadata = {
                k: v for k, v in sorted(
                    self.metadata.items(),
                    key=lambda x: metadata_tags_order.get(x[0], 999)
                )
            }

        self.alias = self.metadata.get('ss_output_name', self.name)

        self.set_hash(
            self.metadata.get('sshs_model_hash') or
            hashes.sha256_from_cache(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or
            ''
        )

        self.sd_version = self.detect_version()

    def detect_version(self) -> SdVersion:
        match (self.metadata.get('ss_base_model_version', ""), self.metadata.get('ss_v2', "")):
            case (version, _) if str(version).startswith("sdxl_"):
                return SdVersion.SDXL
            case (_, "True"):
                return SdVersion.SD2
            case _ if self.metadata:
                return SdVersion.SD1
            case _:
                return SdVersion.Unknown

    def set_hash(self, v: str) -> None:
        self.hash = v
        self.shorthash = self.hash[0:12]

        if self.shorthash:
            import networks
            networks.available_network_hash_lookup[self.shorthash] = self

    def read_hash(self) -> None:
        if not self.hash:
            self.set_hash(hashes.sha256(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or '')

    def get_alias(self) -> str:
        import networks
        if shared.opts.lora_preferred_name == "Filename" or self.alias.lower() in networks.forbidden_network_aliases:
            return self.name
        return self.alias


class Network:
    """LoRA Module implementation."""
    
    def __init__(self, name: str, network_on_disk: NetworkOnDisk) -> None:
        self.name: str = name
        self.network_on_disk: NetworkOnDisk = network_on_disk
        self.te_multiplier: float = 1.0
        self.unet_multiplier: float = 1.0
        self.dyn_dim: Optional[int] = None
        self.modules: dict[str, Any] = {}
        self.bundle_embeddings: dict[str, Any] = {}
        self.mtime: Optional[float] = None
        self.mentioned_name: Optional[str] = None
        """the text that was used to add the network to prompt - can be either name or an alias"""


class ModuleType:
    def create_module(self, net: Network, weights: NetworkWeights) -> Optional[Network]:
        return None


class NetworkModule:
    def __init__(self, net: Network, weights: NetworkWeights) -> None:
        self.network: Network = net
        self.network_key: str = weights.network_key
        self.sd_key: str = weights.sd_key
        self.sd_module = weights.sd_module

        if isinstance(self.sd_module, modules.models.sd3.mmdit.QkvLinear):
            s = self.sd_module.weight.shape
            self.shape = (s[0] // 3, s[1])
        elif hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape
        elif isinstance(self.sd_module, nn.MultiheadAttention):
            self.shape = self.sd_module.out_proj.weight.shape
        else:
            self.shape = None

        self.ops = None
        self.extra_kwargs: dict[str, Any] = {}
        
        match self.sd_module:
            case nn.Conv2d():
                self.ops = F.conv2d
                self.extra_kwargs = {
                    'stride': self.sd_module.stride,
                    'padding': self.sd_module.padding
                }
            case nn.Linear():
                self.ops = F.linear
            case nn.LayerNorm():
                self.ops = F.layer_norm
                self.extra_kwargs = {
                    'normalized_shape': self.sd_module.normalized_shape,
                    'eps': self.sd_module.eps
                }
            case nn.GroupNorm():
                self.ops = F.group_norm
                self.extra_kwargs = {
                    'num_groups': self.sd_module.num_groups,
                    'eps': self.sd_module.eps
                }

        self.dim: Optional[int] = None
        self.bias = weights.w.get("bias")
        self.alpha: Optional[float] = weights.w["alpha"].item() if "alpha" in weights.w else None
        self.scale: Optional[float] = weights.w["scale"].item() if "scale" in weights.w else None
        self.dora_scale = weights.w.get("dora_scale", None)
        self.dora_norm_dims: int = len(self.shape) - 1 if self.shape else 0

    def multiplier(self) -> float:
        return self.network.te_multiplier if 'transformer' in self.sd_key[:20] else self.network.unet_multiplier

    def calc_scale(self) -> float:
        if self.scale is not None:
            return self.scale
        if self.dim is not None and self.alpha is not None:
            return self.alpha / self.dim
        return 1.0

    def apply_weight_decompose(self, updown: torch.Tensor, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_weight = orig_weight.to(updown.dtype)
        dora_scale = self.dora_scale.to(device=orig_weight.device, dtype=updown.dtype)
        updown = updown.to(orig_weight.device)

        merged_scale1 = updown + orig_weight
        merged_scale1_norm = (
            merged_scale1.transpose(0, 1)
            .reshape(merged_scale1.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(merged_scale1.shape[1], *[1] * self.dora_norm_dims)
            .transpose(0, 1)
        )

        dora_merged = merged_scale1 * (dora_scale / merged_scale1_norm)
        return dora_merged - orig_weight

    def finalize_updown(self, updown: torch.Tensor, orig_weight: torch.Tensor, 
                       output_shape: tuple, ex_bias: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=updown.dtype)
            updown = updown.reshape(output_shape)

        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)

        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)

        if ex_bias is not None:
            ex_bias = ex_bias * self.multiplier()

        updown = updown * self.calc_scale()

        if self.dora_scale is not None:
            updown = self.apply_weight_decompose(updown, orig_weight)

        return updown * self.multiplier(), ex_bias

    def calc_updown(self, target: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """A general forward implementation for all modules."""
        if self.ops is None:
            raise NotImplementedError()
        
        updown, ex_bias = self.calc_updown(self.sd_module.weight)
        return y + self.ops(x, weight=updown, bias=ex_bias, **self.extra_kwargs)