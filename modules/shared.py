import os
import sys

import gradio as gr

from modules import shared_cmd_options, shared_gradio_themes, options, shared_items, sd_models_types
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401
from modules import util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules import shared_state, styles, interrogate, shared_total_tqdm, memmon

cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

batch_cond_uncond = True  # old field, unused now in favor of shared.opts.batch_cond_uncond
parallel_processing_allowed = True
styles_filename = cmd_opts.styles_file = cmd_opts.styles_file if len(cmd_opts.styles_file) > 0 else [os.path.join(data_path, 'styles.csv')]
config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

demo: gr.Blocks | None = None  # Explicitly allowing None for clarity

device: str | None = None  # Explicitly allowing None for clarity

weight_load_location: str | None = None  # Explicitly allowing None for clarity

xformers_available = False

hypernetworks: dict = {}

loaded_hypernetworks: list = []

state: 'shared_state.State' | None = None  # Explicitly allowing None for clarity

prompt_styles: 'styles.StyleDatabase' | None = None  # Explicitly allowing None for clarity

interrogator: 'interrogate.InterrogateModels' | None = None  # Explicitly allowing None for clarity

face_restorers: list = []

options_templates: dict | None = None  # Explicitly allowing None for clarity
opts: options.Options | None = None  # Explicitly allowing None for clarity
restricted_opts: set[str] | None = None  # Explicitly allowing None for clarity

sd_model: sd_models_types.WebuiSdModel | None = None  # Explicitly allowing None for clarity

settings_components: dict | None = None  # Explicitly allowing None for clarity
"""assigned from ui.py, a mapping on setting names to gradio components responsible for those settings"""

tab_names: list = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers: list = []

clip_model: str | None = None  # Explicitly allowing None for clarity

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm: 'shared_total_tqdm.TotalTQDM' | None = None  # Explicitly allowing None for clarity

mem_mon: 'memmon.MemUsageMonitor' | None = None  # Explicitly allowing None for clarity

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks

hf_endpoint = os.getenv('HF_ENDPOINT', 'https://huggingface.co')