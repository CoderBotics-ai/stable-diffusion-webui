import os
import re
import sys
import inspect
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gradio as gr

from modules import shared, paths, script_callbacks, extensions, script_loading, scripts_postprocessing, errors, timer, util

topological_sort = util.topological_sort

AlwaysVisible = object()


@dataclass
class MaskBlendArgs:
    current_latent: Any
    nmask: Any
    init_latent: Any
    mask: Any
    blended_latent: Any
    denoiser: Optional[Any] = None
    sigma: Optional[Any] = None
    is_final_blend: bool = False

    def __post_init__(self):
        self.is_final_blend = self.denoiser is None


@dataclass
class PostSampleArgs:
    samples: Any


@dataclass
class PostprocessImageArgs:
    image: Any


@dataclass
class PostProcessMaskOverlayArgs:
    index: int
    mask_for_overlay: Any
    overlay_image: Any


@dataclass
class PostprocessBatchListArgs:
    images: List[Any]


@dataclass
class OnComponent:
    component: gr.blocks.Block


class Script:
    name: Optional[str] = None
    """script's internal name derived from title"""

    section: Optional[str] = None
    """name of UI section that the script's controls will be placed into"""

    filename: Optional[str] = None
    args_from: Optional[int] = None
    args_to: Optional[int] = None
    alwayson: bool = False

    is_txt2img: bool = False
    is_img2img: bool = False
    tabname: Optional[str] = None

    group: Optional[Any] = None
    """A gr.Group component that has all script's UI inside it."""

    create_group: bool = True
    """If False, for alwayson scripts, a group component will not be created."""

    infotext_fields: Optional[List[Tuple[Any, str]]] = None
    """if set in ui(), this is a list of pairs of gradio component + text; the text will be used when
    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example
    """

    paste_field_names: Optional[List[str]] = None
    """if set in ui(), this is a list of names of infotext fields; the fields will be sent through the
    various "Send to <X>" buttons when clicked
    """

    api_info: Optional[Any] = None
    """Generated value of type modules.api.models.ScriptInfo with information about the script for API"""

    on_before_component_elem_id: Optional[List[Tuple[str, Callable]]] = None
    """list of callbacks to be called before a component with an elem_id is created"""

    on_after_component_elem_id: Optional[List[Tuple[str, Callable]]] = None
    """list of callbacks to be called after a component with an elem_id is created"""

    setup_for_ui_only: bool = False
    """If true, the script setup will only be run in Gradio UI, not in API"""

    controls: Optional[List[Any]] = None
    """A list of controls returned by the ui()."""

    def title(self) -> str:
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""
        raise NotImplementedError()

    def ui(self, is_img2img: bool) -> Optional[List[Any]]:
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        pass

    def show(self, is_img2img: bool) -> Union[bool, object]:
        """
        is_img2img is True if this function is called for the img2img interface, and False otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
         """
        return True

    def run(self, p: Any, *args: Any) -> Optional[Any]:
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.

        Usually the processing is done by calling the processing.process_images function.

        args contains all values returned by components from ui()
        """
        pass

    def setup(self, p: Any, *args: Any) -> None:
        """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
        args contains all values returned by components from ui().
        """
        pass

    # ... [rest of the methods with similar type annotations]

    def elem_id(self, item_id: str) -> str:
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""
        need_tabname = self.show(True) == self.show(False)
        tabkind = 'img2img' if self.is_img2img else 'txt2img'
        tabname = f"{tabkind}_" if need_tabname else ""
        title = re.sub(r'[^a-z_0-9]', '', re.sub(r'\s', '_', self.title().lower()))

        return f'script_{tabname}{title}_{item_id}'

# ... [rest of the code with similar improvements]