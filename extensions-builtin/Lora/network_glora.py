from typing import Optional
import network

class ModuleTypeGLora(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        """Create a GLora network module if the required weights are present.
        
        Args:
            net: The network instance
            weights: The network weights containing GLora parameters
            
        Returns:
            NetworkModuleGLora instance if required weights exist, None otherwise
        """
        required_weights = {"a1.weight", "a2.weight", "alpha", "b1.weight", "b2.weight"}
        if required_weights.issubset(weights.w.keys()):
            return NetworkModuleGLora(net, weights)
        
        return None

# adapted from https://github.com/KohakuBlueleaf/LyCORIS
class NetworkModuleGLora(network.NetworkModule):
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        """Initialize GLora network module.
        
        Args:
            net: The network instance
            weights: The network weights containing GLora parameters
        """
        super().__init__(net, weights)

        # Store shape if module has weight attribute
        self.shape = getattr(self.sd_module, 'weight').shape if hasattr(self.sd_module, 'weight') else None

        # Load required weights
        self.w1a = weights.w["a1.weight"]
        self.w1b = weights.w["b1.weight"]
        self.w2a = weights.w["a2.weight"]
        self.w2b = weights.w["b2.weight"]

    def calc_updown(self, orig_weight) -> network.NetworkModule:
        """Calculate the up-down weights for GLora.
        
        Args:
            orig_weight: Original weight tensor
            
        Returns:
            Processed network module with updated weights
        """
        # Move weights to appropriate device
        w1a = self.w1a.to(orig_weight.device)
        w1b = self.w1b.to(orig_weight.device)
        w2a = self.w2a.to(orig_weight.device)
        w2b = self.w2b.to(orig_weight.device)

        output_shape = [w1a.size(0), w1b.size(1)]
        
        # Calculate updown weights using matrix operations
        updown = ((w2b @ w1b) + ((orig_weight.to(dtype=w1a.dtype) @ w2a) @ w1a))

        return self.finalize_updown(updown, orig_weight, output_shape)