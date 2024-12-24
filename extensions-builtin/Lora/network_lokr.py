from typing import Optional
import torch

import lyco_helpers
import network


class ModuleTypeLokr(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        """
        Create a LoKr network module if the required weights are present.
        
        Args:
            net: The network instance
            weights: Network weights containing LoKr parameters
            
        Returns:
            NetworkModuleLokr instance if required weights exist, None otherwise
        """
        has_1: bool = "lokr_w1" in weights.w or ("lokr_w1_a" in weights.w and "lokr_w1_b" in weights.w)
        has_2: bool = "lokr_w2" in weights.w or ("lokr_w2_a" in weights.w and "lokr_w2_b" in weights.w)
        
        if has_1 and has_2:
            return NetworkModuleLokr(net, weights)

        return None


def make_kron(orig_shape: tuple[int, ...], w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    Compute Kronecker product and reshape to original shape.
    
    Args:
        orig_shape: Target shape for the output tensor
        w1: First input tensor
        w2: Second input tensor
        
    Returns:
        Kronecker product reshaped to orig_shape
    """
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(orig_shape)


class NetworkModuleLokr(network.NetworkModule):
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        super().__init__(net, weights)

        # Initialize weight components
        self.w1: Optional[torch.Tensor] = weights.w.get("lokr_w1")
        self.w1a: Optional[torch.Tensor] = weights.w.get("lokr_w1_a")
        self.w1b: Optional[torch.Tensor] = weights.w.get("lokr_w1_b")
        self.dim: int = self.w1b.shape[0] if self.w1b is not None else self.dim
        
        self.w2: Optional[torch.Tensor] = weights.w.get("lokr_w2")
        self.w2a: Optional[torch.Tensor] = weights.w.get("lokr_w2_a")
        self.w2b: Optional[torch.Tensor] = weights.w.get("lokr_w2_b")
        self.dim = self.w2b.shape[0] if self.w2b is not None else self.dim
        self.t2: Optional[torch.Tensor] = weights.w.get("lokr_t2")

    def calc_updown(self, orig_weight: torch.Tensor) -> torch.Tensor:
        """
        Calculate the up-down weights using LoKr components.
        
        Args:
            orig_weight: Original weight tensor
            
        Returns:
            Calculated up-down weights
        """
        # Calculate w1 component
        if self.w1 is not None:
            w1 = self.w1.to(orig_weight.device)
        else:
            w1a = self.w1a.to(orig_weight.device)
            w1b = self.w1b.to(orig_weight.device)
            w1 = w1a @ w1b

        # Calculate w2 component
        match (self.w2, self.t2):
            case (tensor, None) if tensor is not None:
                w2 = tensor.to(orig_weight.device)
            case (None, None):
                w2a = self.w2a.to(orig_weight.device)
                w2b = self.w2b.to(orig_weight.device)
                w2 = w2a @ w2b
            case (None, t2):
                t2 = t2.to(orig_weight.device)
                w2a = self.w2a.to(orig_weight.device)
                w2b = self.w2b.to(orig_weight.device)
                w2 = lyco_helpers.make_weight_cp(t2, w2a, w2b)

        output_shape = [w1.size(0) * w2.size(0), w1.size(1) * w2.size(1)]
        if len(orig_weight.shape) == 4:
            output_shape = orig_weight.shape

        updown = make_kron(output_shape, w1, w2)

        return self.finalize_updown(updown, orig_weight, output_shape)