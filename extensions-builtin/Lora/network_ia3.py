from typing import Optional
import network
from dataclasses import dataclass


@dataclass
class ModuleTypeIa3(network.ModuleType):
    """Module type implementation for IA3 network modules."""
    
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        """
        Create an IA3 network module if the weights contain the required components.
        
        Args:
            net: The network instance
            weights: The network weights containing the required components
            
        Returns:
            NetworkModuleIa3 instance if weights match requirements, None otherwise
        """
        match weights.w:
            case {"weight": _}:
                return NetworkModuleIa3(net, weights)
            case _:
                return None


class NetworkModuleIa3(network.NetworkModule):
    """Implementation of IA3 network module."""
    
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        """
        Initialize the IA3 network module.
        
        Args:
            net: The network instance
            weights: The network weights containing weight and on_input parameters
        """
        super().__init__(net, weights)
        
        self.w = weights.w["weight"]
        self.on_input = weights.w["on_input"].item()

    def calc_updown(self, orig_weight) -> network.NetworkModule:
        """
        Calculate the up/down scaling based on original weights.
        
        Args:
            orig_weight: Original weight tensor
            
        Returns:
            Processed network module with updated weights
        """
        w = self.w.to(orig_weight.device)

        output_shape = [w.size(0), orig_weight.size(1)]
        if self.on_input:
            output_shape.reverse()
        else:
            w = w.reshape(-1, 1)

        updown = orig_weight * w

        return self.finalize_updown(updown, orig_weight, output_shape)