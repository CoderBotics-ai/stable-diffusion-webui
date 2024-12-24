import datetime
import glob
import html
import os
import inspect
from collections import deque
from contextlib import closing
from statistics import stdev, mean
from typing import Optional, Dict, List, Tuple, Any, Union

import modules.textual_inversion.dataset
import torch
import tqdm
from einops import rearrange, repeat
from ldm.util import default
from modules import devices, sd_models, shared, sd_samplers, hashes, sd_hijack_checkpoint, errors
from modules.textual_inversion import textual_inversion, saving_settings
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from torch import einsum
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, zeros_

# Create optimizer dictionary using dictionary comprehension
optimizer_dict: Dict[str, Any] = {
    optim_name: cls_obj 
    for optim_name, cls_obj in inspect.getmembers(torch.optim, inspect.isclass) 
    if optim_name != "Optimizer"
}

class HypernetworkModule(torch.nn.Module):
    activation_dict: Dict[str, Any] = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    activation_dict.update({
        cls_name.lower(): cls_obj 
        for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation) 
        if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'
    })

    def __init__(
        self, 
        dim: int, 
        state_dict: Optional[Dict] = None, 
        layer_structure: Optional[List[float]] = None,
        activation_func: Optional[str] = None,
        weight_init: str = 'Normal',
        add_layer_norm: bool = False,
        activate_output: bool = False,
        dropout_structure: Optional[List[float]] = None
    ) -> None:
        super().__init__()

        self.multiplier: float = 1.0

        if layer_structure is None:
            raise ValueError("layer_structure must not be None")
        if layer_structure[0] != 1:
            raise ValueError("Multiplier Sequence should start with size 1!")
        if layer_structure[-1] != 1:
            raise ValueError("Multiplier Sequence should end with size 1!")

        linears: List[torch.nn.Module] = []
        for i in range(len(layer_structure) - 1):
            # Add a fully-connected layer
            linears.append(
                torch.nn.Linear(
                    int(dim * layer_structure[i]), 
                    int(dim * layer_structure[i+1])
                )
            )

            # Add activation function except last layer
            if not (activation_func == "linear" or activation_func is None or 
                   (i >= len(layer_structure) - 2 and not activate_output)):
                if activation_func not in self.activation_dict:
                    raise ValueError(f'hypernetwork uses an unsupported activation function: {activation_func}')
                linears.append(self.activation_dict[activation_func]())

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            # Add dropout if specified
            if dropout_structure is not None and dropout_structure[i+1] > 0:
                if not 0 < dropout_structure[i+1] < 1:
                    raise ValueError("Dropout probability should be 0 or float between 0 and 1!")
                linears.append(torch.nn.Dropout(p=dropout_structure[i+1]))

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            self._initialize_weights(weight_init, activation_func)

        devices.torch_npu_set_device()
        self.to(devices.device)

    def _initialize_weights(self, weight_init: str, activation_func: Optional[str]) -> None:
        for layer in self.linear:
            if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm)):
                w, b = layer.weight.data, layer.bias.data
                if weight_init == "Normal" or isinstance(layer, torch.nn.LayerNorm):
                    normal_(w, mean=0.0, std=0.01)
                    normal_(b, mean=0.0, std=0)
                elif weight_init == 'XavierUniform':
                    xavier_uniform_(w)
                    zeros_(b)
                elif weight_init == 'XavierNormal':
                    xavier_normal_(w)
                    zeros_(b)
                elif weight_init == 'KaimingUniform':
                    nonlinearity = 'leaky_relu' if activation_func == 'leakyrelu' else 'relu'
                    kaiming_uniform_(w, nonlinearity=nonlinearity)
                    zeros_(b)
                elif weight_init == 'KaimingNormal':
                    nonlinearity = 'leaky_relu' if activation_func == 'leakyrelu' else 'relu'
                    kaiming_normal_(w, nonlinearity=nonlinearity)
                    zeros_(b)
                else:
                    raise ValueError(f"Key {weight_init} is not defined as initialization!")

    def fix_old_state_dict(self, state_dict: Dict[str, Any]) -> None:
        changes = {
            'linear1.bias': 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias': 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        for fr, to in changes.items():
            if (x := state_dict.get(fr)) is not None:
                del state_dict[fr]
                state_dict[to] = x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x) * (self.multiplier if not self.training else 1)

    def trainables(self) -> List[torch.Tensor]:
        return [
            param for layer in self.linear 
            if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm))
            for param in (layer.weight, layer.bias)
        ]

[... rest of the file continues similarly with type hints and modern syntax ...]