from typing import Optional
import network


class ModuleTypeIa3(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional[network.NetworkModule]:
        """Create an IA3 network module if the weights contain the required components.

        Args:
            net: The network instance
            weights: The network weights containing the required components

        Returns:
            NetworkModuleIa3 instance if weights contain required components, None otherwise
        """
        match weights.w:
            case {"weight": _}:
                return NetworkModuleIa3(net, weights)
            case _:
                return None


class NetworkModuleIa3(network.NetworkModule):
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        """Initialize the IA3 network module.

        Args:
            net: The network instance
            weights: The network weights containing the required components
        """
        super().__init__(net, weights)

        self.w = weights.w["weight"]
        self.on_input = weights.w["on_input"].item()

    def calc_updown(self, orig_weight) -> network.NetworkWeights:
        """Calculate the up-down weights for the IA3 module.

        Args:
            orig_weight: The original weight tensor

        Returns:
            The calculated up-down weights
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