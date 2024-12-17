from typing import Optional
import network
from dataclasses import dataclass


@dataclass
class ModuleTypeNorm(network.ModuleType):
    """Module type for normalization operations."""
    
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        """
        Create a normalization module if the required weights are present.
        
        Args:
            net: The network instance
            weights: The network weights
            
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
            weights: The network weights
        """
        super().__init__(net, weights)

        self._w_norm = weights.w.get("w_norm")
        self._b_norm = weights.w.get("b_norm")

    @property
    def w_norm(self):
        """Get the normalization weights."""
        return self._w_norm

    @property
    def b_norm(self):
        """Get the normalization bias."""
        return self._b_norm

    def calc_updown(self, orig_weight) -> tuple:
        """
        Calculate the up/down normalization.
        
        Args:
            orig_weight: The original weight tensor
            
        Returns:
            Tuple containing the normalized results
        """
        output_shape = self._w_norm.shape
        updown = self._w_norm.to(orig_weight.device)

        if self._b_norm is not None:
            ex_bias = self._b_norm.to(orig_weight.device)
        else:
            ex_bias = None

        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias)