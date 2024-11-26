"""
Shared global variables and configuration settings for Stable Diffusion Web UI.

This module contains shared state, configuration options, and global variables used
throughout the application. It serves as a central point for accessing common
resources and settings.
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import gradio as gr

from modules import (
    shared_cmd_options,
    shared_gradio_themes,
    options,
    shared_items,
    sd_models_types,
    util
)
from modules.paths_internal import (
    models_path, script_path, data_path, sd_configs_path,
    sd_default_config, sd_model_file, default_sd_model_file,
    extensions_dir, extensions_builtin_dir
)

if TYPE_CHECKING:
    from modules import shared_state, styles, interrogate, shared_total_tqdm, memmon

# Command line options and configuration
cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
styles_filename = cmd_opts.styles_file if cmd_opts.styles_file else [os.path.join(data_path, 'styles.csv')]

# Gradio UI related
demo: Optional[gr.Blocks] = None
settings_components: Dict[str, Any] = None  # Mapping of setting names to gradio components
tab_names: List[str] = []
gradio_theme = gr.themes.Base()

# Model and processing related
device: Optional[str] = None
weight_load_location: Optional[str] = None
sd_model: Optional[sd_models_types.WebuiSdModel] = None
clip_model = None

# Feature flags and capabilities
batch_cond_uncond: bool = True  # Deprecated: use shared.opts.batch_cond_uncond instead
parallel_processing_allowed: bool = True
xformers_available: bool = False

# Networks and models
hypernetworks: Dict[str, Any] = {}
loaded_hypernetworks: List[Any] = []
face_restorers: List[Any] = []
sd_upscalers: List[Any] = []

# State and runtime objects
state: Optional['shared_state.State'] = None
prompt_styles: Optional['styles.StyleDatabase'] = None
interrogator: Optional['interrogate.InterrogateModels'] = None
progress_print_out = sys.stdout
total_tqdm: Optional['shared_total_tqdm.TotalTQDM'] = None
mem_mon: Optional['memmon.MemUsageMonitor'] = None

# Options and settings
options_templates: Optional[Dict] = None
opts: Optional[options.Options] = None
restricted_opts: Optional[Set[str]] = None

class LatentUpscaleMode(Enum):
    """Enumeration of available latent upscale modes with their configurations."""
    LATENT = {"mode": "bilinear", "antialias": False}
    LATENT_ANTIALIASED = {"mode": "bilinear", "antialias": True}
    LATENT_BICUBIC = {"mode": "bicubic", "antialias": False}
    LATENT_BICUBIC_ANTIALIASED = {"mode": "bicubic", "antialias": True}
    LATENT_NEAREST = {"mode": "nearest", "antialias": False}
    LATENT_NEAREST_EXACT = {"mode": "nearest-exact", "antialias": False}

# Latent upscale configuration
latent_upscale_default_mode: str = "Latent"
latent_upscale_modes: Dict[str, Dict[str, bool]] = {
    "Latent": LatentUpscaleMode.LATENT.value,
    "Latent (antialiased)": LatentUpscaleMode.LATENT_ANTIALIASED.value,
    "Latent (bicubic)": LatentUpscaleMode.LATENT_BICUBIC.value,
    "Latent (bicubic antialiased)": LatentUpscaleMode.LATENT_BICUBIC_ANTIALIASED.value,
    "Latent (nearest)": LatentUpscaleMode.LATENT_NEAREST.value,
    "Latent (nearest-exact)": LatentUpscaleMode.LATENT_NEAREST_EXACT.value,
}

# Options helpers
options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

# Utility functions
natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

# Theme and UI functions
reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

# Shared item functions
list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks

# External service configuration
hf_endpoint: str = os.getenv('HF_ENDPOINT', 'https://huggingface.co')