from __future__ import annotations

import functools
import logging
from typing import Dict, List, Optional, Tuple, Set

from modules import (
    sd_samplers_kdiffusion,
    sd_samplers_timesteps,
    sd_samplers_lcm,
    shared,
    sd_samplers_common,
    sd_schedulers,
)

# Re-export commonly used functions
samples_to_image_grid = sd_samplers_common.samples_to_image_grid
sample_to_image = sd_samplers_common.sample_to_image

# Initialize available samplers
all_samplers: List[sd_samplers_common.SamplerData] = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_timesteps.samplers_data_timesteps,
    *sd_samplers_lcm.samplers_data_lcm,
]
all_samplers_map: Dict[str, sd_samplers_common.SamplerData] = {x.name: x for x in all_samplers}

# Global sampler state
samplers: List[sd_samplers_common.SamplerData] = []
samplers_for_img2img: List[sd_samplers_common.SamplerData] = []
samplers_map: Dict[str, str] = {}
samplers_hidden: Set[str] = set()


def find_sampler_config(name: Optional[str]) -> Optional[sd_samplers_common.SamplerData]:
    """
    Find sampler configuration by name.
    
    Args:
        name: Name of the sampler to find
        
    Returns:
        Sampler configuration if found, otherwise returns first available sampler
    """
    if name is not None:
        config = all_samplers_map.get(name)
    else:
        config = all_samplers[0]

    return config


def create_sampler(name: str, model: any) -> sd_samplers_common.Sampler:
    """
    Create a sampler instance with the given name and model.
    
    Args:
        name: Name of the sampler to create
        model: Model instance to use with the sampler
        
    Returns:
        Initialized sampler instance
        
    Raises:
        AssertionError: If sampler name is invalid
        Exception: If sampler is not supported for SDXL
    """
    config = find_sampler_config(name)

    if config is None:
        raise AssertionError(f'Invalid sampler name: {name}')

    if model.is_sdxl and config.options.get("no_sdxl", False):
        raise Exception(f"Sampler {config.name} is not supported for SDXL")

    sampler = config.constructor(model)
    sampler.config = config

    return sampler


def set_samplers() -> None:
    """Initialize global sampler configurations."""
    global samplers, samplers_for_img2img, samplers_hidden

    samplers_hidden = set(shared.opts.hide_samplers)
    samplers = all_samplers
    samplers_for_img2img = all_samplers

    # Update sampler name mappings
    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name


def visible_sampler_names() -> List[str]:
    """Get names of all visible (non-hidden) samplers."""
    return [x.name for x in samplers if x.name not in samplers_hidden]


def visible_samplers() -> List[sd_samplers_common.SamplerData]:
    """Get configurations for all visible (non-hidden) samplers."""
    return [x for x in samplers if x.name not in samplers_hidden]


def get_sampler_from_infotext(info: Dict[str, str]) -> str:
    """Extract sampler name from generation info text."""
    return get_sampler_and_scheduler(info.get("Sampler"), info.get("Schedule type"))[0]


def get_scheduler_from_infotext(info: Dict[str, str]) -> str:
    """Extract scheduler name from generation info text."""
    return get_sampler_and_scheduler(info.get("Sampler"), info.get("Schedule type"))[1]


def get_hr_sampler_and_scheduler(info: Dict[str, str]) -> Tuple[str, str]:
    """
    Get high-res fix sampler and scheduler configuration from info text.
    
    Args:
        info: Generation parameters dictionary
        
    Returns:
        Tuple of (sampler name, scheduler name)
    """
    hr_sampler = info.get("Hires sampler", "Use same sampler")
    sampler = info.get("Sampler") if hr_sampler == "Use same sampler" else hr_sampler

    hr_scheduler = info.get("Hires schedule type", "Use same scheduler") 
    scheduler = info.get("Schedule type") if hr_scheduler == "Use same scheduler" else hr_scheduler

    sampler, scheduler = get_sampler_and_scheduler(sampler, scheduler)

    # Preserve special "Use same" values if unchanged
    sampler = sampler if sampler != info.get("Sampler") else "Use same sampler"
    scheduler = scheduler if scheduler != info.get("Schedule type") else "Use same scheduler"

    return sampler, scheduler


def get_hr_sampler_from_infotext(info: Dict[str, str]) -> str:
    """Get high-res fix sampler name from info text."""
    return get_hr_sampler_and_scheduler(info)[0]


def get_hr_scheduler_from_infotext(info: Dict[str, str]) -> str:
    """Get high-res fix scheduler name from info text."""
    return get_hr_sampler_and_scheduler(info)[1]


@functools.cache
def get_sampler_and_scheduler(
    sampler_name: Optional[str],
    scheduler_name: Optional[str],
    *,
    convert_automatic: bool = True
) -> Tuple[str, str]:
    """
    Get validated sampler and scheduler names, with fallbacks to defaults.
    
    Args:
        sampler_name: Requested sampler name
        scheduler_name: Requested scheduler name
        convert_automatic: Whether to convert to automatic scheduler if it's the default
        
    Returns:
        Tuple of (sampler name, scheduler label)
    """
    default_sampler = samplers[0]
    found_scheduler = sd_schedulers.schedulers_map.get(
        scheduler_name,
        sd_schedulers.schedulers[0]
    )

    name = sampler_name or default_sampler.name

    # Extract scheduler from combined name if present
    for scheduler in sd_schedulers.schedulers:
        name_options = [
            scheduler.label,
            scheduler.name,
            *(scheduler.aliases or [])
        ]

        for name_option in name_options:
            if name.endswith(f" {name_option}"):
                found_scheduler = scheduler
                name = name[0:-(len(name_option) + 1)]
                break

    sampler = all_samplers_map.get(name, default_sampler)

    # Use automatic scheduler if it's the default for this sampler
    if (convert_automatic and 
        sampler.options.get('scheduler') == found_scheduler.name):
        found_scheduler = sd_schedulers.schedulers[0]

    return sampler.name, found_scheduler.label


def fix_p_invalid_sampler_and_scheduler(p: any) -> None:
    """
    Validate and correct invalid sampler/scheduler combinations.
    
    Args:
        p: Processing object with sampler_name and scheduler attributes
    """
    i_sampler_name, i_scheduler = p.sampler_name, p.scheduler
    p.sampler_name, p.scheduler = get_sampler_and_scheduler(
        p.sampler_name,
        p.scheduler,
        convert_automatic=False
    )
    
    if p.sampler_name != i_sampler_name or i_scheduler != p.scheduler:
        logging.warning(
            f'Sampler Scheduler autocorrection: '
            f'"{i_sampler_name}" -> "{p.sampler_name}", '
            f'"{i_scheduler}" -> "{p.scheduler}"'
        )


# Initialize samplers on module load
set_samplers()