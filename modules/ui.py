"""
Main UI module for Stable Diffusion Web UI.
Handles creation and setup of the web interface components using Gradio.
"""

import datetime
import mimetypes
import os
import sys
from functools import reduce
import warnings
from contextlib import ExitStack
from typing import List, Dict, Any, Optional, Tuple, Callable

import gradio as gr
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin

from modules import (
    gradio_extensons, sd_schedulers, sd_hijack, sd_models,
    script_callbacks, ui_extensions, deepbooru, extra_networks,
    ui_common, ui_postprocessing, progress, ui_loadsave,
    shared_items, ui_settings, timer, sysinfo, ui_checkpoint_merger,
    scripts, sd_samplers, processing, ui_extra_networks, ui_toprow,
    launch_utils
)
from modules.ui_components import (
    FormRow, FormGroup, ToolButton, FormHTML,
    InputAccordion, ResizeHandleRow
)
from modules.paths import script_path
from modules.ui_common import create_refresh_button
from modules.ui_gradio_extensions import reload_javascript
from modules.shared import opts, cmd_opts
import modules.infotext_utils as parameters_copypaste
import modules.hypernetworks.ui as hypernetworks_ui
import modules.textual_inversion.ui as textual_inversion_ui
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.shared as shared
from modules import prompt_parser
from modules.sd_hijack import model_hijack
from modules.infotext_utils import image_from_url_text, PasteField

# Constants
RANDOM_SYMBOL = '\U0001f3b2\ufe0f'  # 🎲️
REUSE_SYMBOL = '\u267b\ufe0f'  # ♻️
PASTE_SYMBOL = '\u2199\ufe0f'  # ↙
REFRESH_SYMBOL = '\U0001f504'  # 🔄
SAVE_STYLE_SYMBOL = '\U0001f4be'  # 💾
APPLY_STYLE_SYMBOL = '\U0001f4cb'  # 📋
CLEAR_PROMPT_SYMBOL = '\U0001f5d1\ufe0f'  # 🗑️
EXTRA_NETWORKS_SYMBOL = '\U0001F3B4'  # 🎴
SWITCH_VALUES_SYMBOL = '\U000021C5' # ⇅
RESTORE_PROGRESS_SYMBOL = '\U0001F300' # 🌀
DETECT_IMAGE_SIZE_SYMBOL = '\U0001F4D0'  # 📐

# Initialize mimetypes
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')
mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('image/avif', '.avif')

# Disable Gradio analytics if not sharing
if not cmd_opts.share and not cmd_opts.listen:
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

# Setup ngrok if configured
if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok
    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_options
    )

def gr_show(visible: bool = True) -> Dict[str, Any]:
    """Helper function to show/hide Gradio components."""
    return {"visible": visible, "__type__": "update"}

def send_gradio_gallery_to_image(x: List[str]) -> Optional[Image.Image]:
    """Convert Gradio gallery output to PIL Image."""
    if not x:
        return None
    return image_from_url_text(x[0])

def calc_resolution_hires(
    enable: bool,
    width: int,
    height: int,
    hr_scale: float,
    hr_resize_x: int,
    hr_resize_y: int
) -> str:
    """Calculate high-res resolution string."""
    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(
        width=width,
        height=height,
        enable_hr=True,
        hr_scale=hr_scale,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y
    )
    p.calculate_target_resolution()

    return (
        f"from <span class='resolution'>{p.width}x{p.height}</span> "
        f"to <span class='resolution'>"
        f"{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}"
        f"</span>"
    )

def resize_from_to_html(width: int, height: int, scale_by: float) -> str:
    """Generate HTML for resize information."""
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)

    if not target_width or not target_height:
        return "no image selected"

    return (
        f"resize: from <span class='resolution'>{width}x{height}</span> "
        f"to <span class='resolution'>{target_width}x{target_height}</span>"
    )

def process_interrogate(
    interrogation_function: Callable,
    mode: int,
    ii_input_dir: str,
    ii_output_dir: str,
    *ii_singles: Any
) -> Tuple[Optional[str], None]:
    """Process image interrogation requests."""
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    elif mode == 5:
        assert not shared.cmd_opts.hide_ui_dir_config, \
            "Launched with --hide-ui-dir-config, batch img2img disabled"
        
        images = shared.listfiles(ii_input_dir)
        print(f"Will process {len(images)} images.")
        
        if ii_output_dir:
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir

        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            with open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8') as f:
                print(interrogation_function(img), file=f)

        return [gr.update(), None]

def interrogate(image: Image.Image) -> Optional[str]:
    """Run CLIP interrogation on image."""
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt

def interrogate_deepbooru(image: Image.Image) -> Optional[str]:
    """Run DeepBooru interrogation on image."""
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt

def connect_clear_prompt(button: gr.Button) -> None:
    """Setup clear prompt button click event."""
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )

# Main UI creation functions continue below...
# (The rest of the file would continue with similar improvements)