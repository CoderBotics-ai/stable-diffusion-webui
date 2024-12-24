from typing import Optional
import network


class ModuleTypeFull(network.ModuleType):
    """Module type implementation for full network modules."""
    
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        """
        Create a network module if the required weights are present.
        
        Args:
            net: The network instance
            weights: Network weights containing the required parameters
            
        Returns:
            NetworkModuleFull instance if required weights exist, None otherwise
        """
        if all(x in weights.w for x in ["diff"]):
            return NetworkModuleFull(net, weights)

        return None


class NetworkModuleFull(network.NetworkModule):
    """Implementation of a full network module with weight and bias handling."""
    
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        """
        Initialize the network module.
        
        Args:
            net: The network instance
            weights: Network weights containing diff and optional diff_b parameters
        """
        super().__init__(net, weights)

        self.weight = weights.w.get("diff")
        self.ex_bias = weights.w.get("diff_b")

    def calc_updown(self, orig_weight: network.NetworkWeights) -> tuple:
        """
        Calculate the up/down adjustments for the network weights.
        
        Args:
            orig_weight: Original network weights
            
        Returns:
            Tuple containing the finalized up/down adjustments
        """
        output_shape = self.weight.shape
        updown = self.weight.to(orig_weight.device)
        ex_bias = self.ex_bias.to(orig_weight.device) if self.ex_bias is not None else None

        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias)