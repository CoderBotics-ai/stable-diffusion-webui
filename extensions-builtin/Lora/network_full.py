from typing import Optional
import network
from dataclasses import dataclass


@dataclass
class ModuleTypeFull(network.ModuleType):
    """Module type implementation for full network modules."""
    
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        match weights.w:
            case {"diff": _}:
                return NetworkModuleFull(net, weights)
            case _:
                return None


class NetworkModuleFull(network.NetworkModule):
    """Network module implementation for full network calculations."""
    
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        super().__init__(net, weights)

        # Using pattern matching for dictionary access (Python 3.12 feature)
        match weights.w:
            case {"diff": weight, "diff_b": bias}:
                self.weight = weight
                self.ex_bias = bias
            case {"diff": weight}:
                self.weight = weight
                self.ex_bias = None
            case _:
                raise ValueError("Invalid weights configuration")

    def calc_updown(self, orig_weight) -> tuple:
        """Calculate up/down values for network weights.
        
        Args:
            orig_weight: Original weight tensor
            
        Returns:
            Tuple containing finalized up/down calculations
        """
        output_shape = self.weight.shape
        updown = self.weight.to(orig_weight.device)
        ex_bias = self.ex_bias.to(orig_weight.device) if self.ex_bias is not None else None

        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias)