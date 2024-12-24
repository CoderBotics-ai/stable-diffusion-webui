from typing import Optional
import network


class ModuleTypeNorm(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        """
        Create a normalization module if the required weights are present.
        
        Args:
            net: The network instance
            weights: Network weights containing normalization parameters
            
        Returns:
            NetworkModuleNorm instance if required weights exist, None otherwise
        """
        if all(x in weights.w for x in ["w_norm", "b_norm"]):
            return NetworkModuleNorm(net, weights)

        return None


class NetworkModuleNorm(network.NetworkModule):
    """Network module implementation for normalization operations."""
    
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        """
        Initialize the normalization module.
        
        Args:
            net: The network instance
            weights: Network weights containing normalization parameters
        """
        super().__init__(net, weights)

        self.w_norm = weights.w.get("w_norm")
        self.b_norm = weights.w.get("b_norm")

    def calc_updown(self, orig_weight: network.NetworkWeights) -> tuple:
        """
        Calculate the up/down scaling factors for normalization.
        
        Args:
            orig_weight: Original network weights
            
        Returns:
            Tuple containing the finalized updown parameters
        """
        output_shape = self.w_norm.shape
        updown = self.w_norm.to(orig_weight.device)

        ex_bias: Optional[network.NetworkWeights] = (
            self.b_norm.to(orig_weight.device) if self.b_norm is not None else None
        )

        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias)