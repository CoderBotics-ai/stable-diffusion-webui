"""
Shared configuration and state management module for Stable Diffusion Web UI.

This module contains global configuration, models and state that is shared across
the entire application. It provides centralized access to important components
like models, options, and UI elements.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import gradio as gr

from modules import (
    shared_cmd_options, shared_gradio_themes, options, 
    shared_items, sd_models_types, util
)
from modules.paths_internal import (
    models_path, script_path, data_path, sd_configs_path,
    sd_default_config, sd_model_file, default_sd_model_file,
    extensions_dir, extensions_builtin_dir
)

if TYPE_CHECKING:
    from modules import shared_state, styles, interrogate, shared_total_tqdm, memmon

# Command line options
cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

@dataclass
class UIConfig:
    """Configuration for UI components and settings"""
    styles_filename: List[str]
    config_filename: str
    hide_dirs: Dict[str, bool]
    demo: Optional[gr.Blocks] = None
    settings_components: Dict = None
    tab_names: List[str] = None

    def __post_init__(self):
        self.tab_names = []
        self.settings_components = {}

ui_config = UIConfig(
    styles_filename=cmd_opts.styles_file if cmd_opts.styles_file else [os.path.join(data_path, 'styles.csv')],
    config_filename=cmd_opts.ui_settings_file,
    hide_dirs={"visible": not cmd_opts.hide_ui_dir_config}
)

@dataclass
class ModelConfig:
    """Configuration for ML models and processing"""
    device: Optional[str] = None
    weight_load_location: Optional[str] = None
    xformers_available: bool = False
    batch_cond_uncond: bool = True
    parallel_processing_allowed: bool = True
    
    # Models
    sd_model: Optional[sd_models_types.WebuiSdModel] = None
    clip_model = None
    hypernetworks: Dict = None
    loaded_hypernetworks: List = None
    face_restorers: List = None

model_config = ModelConfig(
    hypernetworks={},
    loaded_hypernetworks=[],
    face_restorers=[]
)

@dataclass
class UpscaleConfig:
    """Configuration for upscaling options"""
    latent_upscale_default_mode: str = "Latent"
    latent_upscale_modes: Dict = None
    sd_upscalers: List = None

    def __post_init__(self):
        self.sd_upscalers = []
        self.latent_upscale_modes = {
            "Latent": {"mode": "bilinear", "antialias": False},
            "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
            "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
            "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
            "Latent (nearest)": {"mode": "nearest", "antialias": False},
            "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
        }

upscale_config = UpscaleConfig()

# State management
state: Optional['shared_state.State'] = None
prompt_styles: Optional['styles.StyleDatabase'] = None
interrogator: Optional['interrogate.InterrogateModels'] = None
total_tqdm: Optional['shared_total_tqdm.TotalTQDM'] = None
mem_mon: Optional['memmon.MemUsageMonitor'] = None

# Options management
options_templates: Dict = None
opts: Optional[options.Options] = None
restricted_opts: Set[str] = set()

# UI Theme
gradio_theme = gr.themes.Base()

# Progress output
progress_print_out = sys.stdout

# Hugging Face endpoint
hf_endpoint = os.getenv('HF_ENDPOINT', 'https://huggingface.co')

# Utility functions
options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

# Import utility functions
natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

# Theme management
reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

# Shared items management
list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks