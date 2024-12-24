from typing import Optional
import network
from dataclasses import dataclass


@dataclass
class ModuleTypeIa3(network.ModuleType):
    """IA3 Module Type implementation for network scaling."""
    
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        """
        Create an IA3 network module if the required weights are present.
        
        Args:
            net: The network instance
            weights: The network weights
            
        Returns:
            NetworkModuleIa3 instance if required weights exist, None otherwise
        """
        match weights.w:
            case {"weight": _, "on_input": _}:
                return NetworkModuleIa3(net, weights)
            case _:
                return None


class NetworkModuleIa3(network.NetworkModule):
    """Implementation of IA3 network module with input-adaptive scaling."""
    
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        """
        Initialize the IA3 network module.
        
        Args:
            net: The network instance
            weights: The network weights containing 'weight' and 'on_input' parameters
        """
        super().__init__(net, weights)
        
        # Validate required weights
        if not all(key in weights.w for key in ("weight", "on_input")):
            raise ValueError("Required weights 'weight' and 'on_input' must be present")
            
        self.w = weights.w["weight"]
        self.on_input = weights.w["on_input"].item()

    def calc_updown(self, orig_weight) -> network.NetworkModule:
        """
        Calculate the scaled weights based on original weights.
        
        Args:
            orig_weight: Original weight tensor
            
        Returns:
            Scaled weight tensor
        """
        w = self.w.to(orig_weight.device)

        output_shape = [w.size(0), orig_weight.size(1)]
        match self.on_input:
            case True:
                output_shape.reverse()
            case False:
                w = w.reshape(-1, 1)

        updown = orig_weight * w

        return self.finalize_updown(updown, orig_weight, output_shape)