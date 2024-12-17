from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from modules import extra_networks, shared
import networks


class ExtraNetworkLora(extra_networks.ExtraNetwork):
    """ExtraNetworkLora handles LORA network operations and management."""

    def __init__(self) -> None:
        super().__init__('lora')
        self.errors: Dict[str, int] = {}
        """mapping of network names to the number of errors the network had during operation"""

    remove_symbols: str = str.maketrans('', '', ":,")

    def activate(self, p: any, params_list: List[extra_networks.ExtraNetworkParams]) -> None:
        """
        Activate LORA networks with the given parameters.

        Args:
            p: The processing object
            params_list: List of network parameters
        """
        additional = shared.opts.sd_lora

        self.errors.clear()

        # Handle additional network if specified
        if (additional != "None" and 
            additional in networks.available_networks and 
            not any(params.items[0] == additional for params in params_list)):
            
            p.all_prompts = [
                f"{prompt}<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>"
                for prompt in p.all_prompts
            ]
            params_list.append(extra_networks.ExtraNetworkParams(
                items=[additional, shared.opts.extra_networks_default_multiplier]
            ))

        names: List[str] = []
        te_multipliers: List[float] = []
        unet_multipliers: List[float] = []
        dyn_dims: List[Optional[int]] = []

        for params in params_list:
            if not params.items:
                continue

            names.append(params.positional[0])

            # Extract and process multipliers and dimensions
            te_multiplier = (float(params.positional[1]) 
                           if len(params.positional) > 1 
                           else 1.0)
            te_multiplier = float(params.named.get("te", te_multiplier))

            unet_multiplier = (float(params.positional[2]) 
                             if len(params.positional) > 2 
                             else te_multiplier)
            unet_multiplier = float(params.named.get("unet", unet_multiplier))

            dyn_dim = (int(params.positional[3]) 
                      if len(params.positional) > 3 
                      else None)
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim

            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            dyn_dims.append(dyn_dim)

        # Load networks with collected parameters
        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)

        # Handle hash generation for infotext
        if shared.opts.lora_add_hashes_to_infotext:
            if not getattr(p, "is_hr_pass", False) or not hasattr(p, "lora_hashes"):
                p.lora_hashes = {}

            for item in networks.loaded_networks:
                if item.network_on_disk.shorthash and item.mentioned_name:
                    cleaned_name = item.mentioned_name.translate(self.remove_symbols)
                    p.lora_hashes[cleaned_name] = item.network_on_disk.shorthash

            if p.lora_hashes:
                p.extra_generation_params["Lora hashes"] = ", ".join(
                    f"{name}: {hash_}" 
                    for name, hash_ in p.lora_hashes.items()
                )

    def deactivate(self, p: any) -> None:
        """
        Deactivate networks and handle any errors that occurred.

        Args:
            p: The processing object
        """
        if self.errors:
            error_messages = ", ".join(f"{name} ({count})" 
                                     for name, count in self.errors.items())
            p.comment(f"Networks with errors: {error_messages}")
            self.errors.clear()