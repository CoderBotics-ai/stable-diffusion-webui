from __future__ import annotations

import json
import os
import re
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any, Pattern
from dataclasses import dataclass

from modules import errors

# Type aliases
ExtraNetworkRegistry = Dict[str, 'ExtraNetwork']
ExtraNetworkAliases = Dict[str, 'ExtraNetwork']

extra_network_registry: ExtraNetworkRegistry = {}
extra_network_aliases: ExtraNetworkAliases = {}

# Compile regex pattern once for better performance
re_extra_net: Pattern[str] = re.compile(r"<(\w+):([^>]+)>")


def initialize() -> None:
    extra_network_registry.clear()
    extra_network_aliases.clear()


def register_extra_network(extra_network: ExtraNetwork) -> None:
    extra_network_registry[extra_network.name] = extra_network


def register_extra_network_alias(extra_network: ExtraNetwork, alias: str) -> None:
    extra_network_aliases[alias] = extra_network


def register_default_extra_networks() -> None:
    from modules.extra_networks_hypernet import ExtraNetworkHypernet
    register_extra_network(ExtraNetworkHypernet())


@dataclass
class ExtraNetworkParams:
    items: List[Any]
    positional: List[Any]
    named: Dict[str, str]

    def __init__(self, items: Optional[List[Any]] = None):
        self.items = items or []
        self.positional = []
        self.named = {}

        for item in self.items:
            if isinstance(item, str):
                parts = item.split('=', 2)
                if len(parts) == 2:
                    self.named[parts[0]] = parts[1]
                else:
                    self.positional.append(item)
            else:
                self.positional.append(item)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtraNetworkParams):
            return NotImplemented
        return self.items == other.items


class ExtraNetwork:
    def __init__(self, name: str):
        self.name = name

    def activate(self, p: Any, params_list: List[ExtraNetworkParams]) -> None:
        """
        Called by processing on every run. Whatever the extra network is meant to do should be activated here.
        Passes arguments related to this extra network in params_list.
        User passes arguments by specifying this in his prompt:

        <name:arg1:arg2:arg3>

        Where name matches the name of this ExtraNetwork object, and arg1:arg2:arg3 are any natural number of text arguments
        separated by colon.

        Even if the user does not mention this ExtraNetwork in his prompt, the call will still be made, with empty params_list -
        in this case, all effects of this extra networks should be disabled.

        Can be called multiple times before deactivate() - each new call should override the previous call completely.
        """
        raise NotImplementedError

    def deactivate(self, p: Any) -> None:
        """
        Called at the end of processing for housekeeping. No need to do anything here.
        """
        raise NotImplementedError


def lookup_extra_networks(extra_network_data: Dict[str, List[ExtraNetworkParams]]) -> Dict[ExtraNetwork, List[ExtraNetworkParams]]:
    """Returns a dict mapping ExtraNetwork objects to lists of arguments for those extra networks."""
    res: Dict[ExtraNetwork, List[ExtraNetworkParams]] = {}

    for extra_network_name, extra_network_args in extra_network_data.items():
        extra_network = extra_network_registry.get(extra_network_name)
        alias = extra_network_aliases.get(extra_network_name)

        if alias is not None and extra_network is None:
            extra_network = alias

        if extra_network is None:
            logging.info(f"Skipping unknown extra network: {extra_network_name}")
            continue

        res.setdefault(extra_network, []).extend(extra_network_args)

    return res


def activate(p: Any, extra_network_data: Dict[str, List[ExtraNetworkParams]]) -> None:
    """Call activate for extra networks in extra_network_data in specified order, then call
    activate for all remaining registered networks with an empty argument list"""
    activated: List[ExtraNetwork] = []

    for extra_network, extra_network_args in lookup_extra_networks(extra_network_data).items():
        try:
            extra_network.activate(p, extra_network_args)
            activated.append(extra_network)
        except Exception as e:
            errors.display(e, f"activating extra network {extra_network.name} with arguments {extra_network_args}")

    for extra_network_name, extra_network in extra_network_registry.items():
        if extra_network in activated:
            continue

        try:
            extra_network.activate(p, [])
        except Exception as e:
            errors.display(e, f"activating extra network {extra_network_name}")

    if p.scripts is not None:
        p.scripts.after_extra_networks_activate(p, batch_number=p.iteration, prompts=p.prompts, 
                                             seeds=p.seeds, subseeds=p.subseeds, 
                                             extra_network_data=extra_network_data)


def deactivate(p: Any, extra_network_data: Dict[str, List[ExtraNetworkParams]]) -> None:
    """Call deactivate for extra networks in extra_network_data in specified order, then call
    deactivate for all remaining registered networks"""
    data = lookup_extra_networks(extra_network_data)

    for extra_network in data:
        try:
            extra_network.deactivate(p)
        except Exception as e:
            errors.display(e, f"deactivating extra network {extra_network.name}")

    for extra_network_name, extra_network in extra_network_registry.items():
        if extra_network in data:
            continue

        try:
            extra_network.deactivate(p)
        except Exception as e:
            errors.display(e, f"deactivating unmentioned extra network {extra_network_name}")


def parse_prompt(prompt: str) -> tuple[str, defaultdict[str, List[ExtraNetworkParams]]]:
    res: defaultdict[str, List[ExtraNetworkParams]] = defaultdict(list)

    def found(m: re.Match[str]) -> str:
        name = m.group(1)
        args = m.group(2)
        res[name].append(ExtraNetworkParams(items=args.split(":")))
        return ""

    prompt = re.sub(re_extra_net, found, prompt)
    return prompt, res


def parse_prompts(prompts: List[str]) -> tuple[List[str], Optional[defaultdict[str, List[ExtraNetworkParams]]]]:
    res: List[str] = []
    extra_data: Optional[defaultdict[str, List[ExtraNetworkParams]]] = None

    for prompt in prompts:
        updated_prompt, parsed_extra_data = parse_prompt(prompt)
        if extra_data is None:
            extra_data = parsed_extra_data
        res.append(updated_prompt)

    return res, extra_data


def get_user_metadata(filename: Optional[str], lister: Optional[Any] = None) -> Dict[str, Any]:
    if filename is None:
        return {}

    basename, ext = os.path.splitext(filename)
    metadata_filename = basename + '.json'

    metadata: Dict[str, Any] = {}
    try:
        exists = lister.exists(metadata_filename) if lister else os.path.exists(metadata_filename)
        if exists:
            with open(metadata_filename, "r", encoding="utf8") as file:
                metadata = json.load(file)
    except Exception as e:
        errors.display(e, f"reading extra network user metadata from {metadata_filename}")

    return metadata