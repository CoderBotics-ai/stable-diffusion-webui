from typing import Optional
import torch
from torch import Tensor

import lyco_helpers
import network


class ModuleTypeLokr(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        """Create a Lokr network module if the required weights are present.

        Args:
            net: The network instance
            weights: The network weights

        Returns:
            NetworkModuleLokr instance if required weights exist, None otherwise
        """
        has_1: bool = "lokr_w1" in weights.w or ("lokr_w1_a" in weights.w and "lokr_w1_b" in weights.w)
        has_2: bool = "lokr_w2" in weights.w or ("lokr_w2_a" in weights.w and "lokr_w2_b" in weights.w)
        
        match (has_1, has_2):
            case (True, True):
                return NetworkModuleLokr(net, weights)
            case _:
                return None


def make_kron(orig_shape: tuple[int, ...], w1: Tensor, w2: Tensor) -> Tensor:
    """Compute Kronecker product and reshape to original shape.

    Args:
        orig_shape: Target shape for the output tensor
        w1: First input tensor
        w2: Second input tensor

    Returns:
        Tensor: Kronecker product reshaped to orig_shape
    """
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(orig_shape)


class NetworkModuleLokr(network.NetworkModule):
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        super().__init__(net, weights)

        # Initialize weights with proper type hints
        self.w1: Optional[Tensor] = weights.w.get("lokr_w1")
        self.w1a: Optional[Tensor] = weights.w.get("lokr_w1_a")
        self.w1b: Optional[Tensor] = weights.w.get("lokr_w1_b")
        self.w2: Optional[Tensor] = weights.w.get("lokr_w2")
        self.w2a: Optional[Tensor] = weights.w.get("lokr_w2_a")
        self.w2b: Optional[Tensor] = weights.w.get("lokr_w2_b")
        self.t2: Optional[Tensor] = weights.w.get("lokr_t2")
        
        # Update dimension based on available weights
        self.dim: int = (self.w1b.shape[0] if self.w1b is not None else 
                        self.w2b.shape[0] if self.w2b is not None else 
                        self.dim)

    def calc_updown(self, orig_weight: Tensor) -> Tensor:
        """Calculate up-down weights using Kronecker products.

        Args:
            orig_weight: Original weight tensor

        Returns:
            Tensor: Calculated up-down weights
        """
        # Calculate w1
        match (self.w1, self.w1a, self.w1b):
            case (None, w1a, w1b) if w1a is not None and w1b is not None:
                w1 = (w1a.to(orig_weight.device) @ 
                      w1b.to(orig_weight.device))
            case (w1, _, _) if w1 is not None:
                w1 = w1.to(orig_weight.device)
            case _:
                raise ValueError("Invalid w1 configuration")

        # Calculate w2
        match (self.w2, self.w2a, self.w2b, self.t2):
            case (w2, _, _, _) if w2 is not None:
                w2 = w2.to(orig_weight.device)
            case (None, w2a, w2b, None) if w2a is not None and w2b is not None:
                w2 = (w2a.to(orig_weight.device) @ 
                      w2b.to(orig_weight.device))
            case (None, w2a, w2b, t2) if all(x is not None for x in (w2a, w2b, t2)):
                w2 = lyco_helpers.make_weight_cp(
                    t2.to(orig_weight.device),
                    w2a.to(orig_weight.device),
                    w2b.to(orig_weight.device)
                )
            case _:
                raise ValueError("Invalid w2 configuration")

        output_shape: list[int] = [w1.size(0) * w2.size(0), w1.size(1) * w2.size(1)]
        if len(orig_weight.shape) == 4:
            output_shape = list(orig_weight.shape)

        updown = make_kron(tuple(output_shape), w1, w2)
        return self.finalize_updown(updown, orig_weight, output_shape)