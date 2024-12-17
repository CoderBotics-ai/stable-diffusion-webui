from __future__ import annotations  # For better type hints support
import network
from typing import Optional


class ModuleTypeFull(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[NetworkModuleFull]:
        """
        Create a network module if the required weights are present.
        
        Args:
            net: The network instance
            weights: The network weights
            
        Returns:
            NetworkModuleFull instance if required weights exist, None otherwise
        """
        if all(x in weights.w for x in ["diff"]):
            return NetworkModuleFull(net, weights)

        return None


class NetworkModuleFull(network.NetworkModule):
    """Network module implementation for full network operations."""
    
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        """
        Initialize the NetworkModuleFull instance.
        
        Args:
            net: The network instance
            weights: The network weights
        """
        super().__init__(net, weights)

        self.weight = weights.w.get("diff")
        self.ex_bias = weights.w.get("diff_b")

    def calc_updown(self, orig_weight) -> tuple:
        """
        Calculate the up/down values for the network.
        
        Args:
            orig_weight: The original weight tensor
            
        Returns:
            Tuple containing the finalized up/down values
        """
        output_shape = self.weight.shape
        updown = self.weight.to(device=orig_weight.device)
        ex_bias = self.ex_bias.to(device=orig_weight.device) if self.ex_bias is not None else None

        return self.finalize_updown(
            updown=updown,
            orig_weight=orig_weight,
            output_shape=output_shape,
            ex_bias=ex_bias
        )