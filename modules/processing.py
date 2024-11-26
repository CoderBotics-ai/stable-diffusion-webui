"""
Core image processing and stable diffusion utilities.
Handles image generation, processing, and sampling for the stable diffusion web UI.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from einops import repeat, rearrange
from PIL import Image, ImageOps
from skimage import exposure

from modules import (
    devices, prompt_parser, masking, sd_samplers, lowvram, 
    infotext_utils, extra_networks, sd_vae_approx, scripts,
    sd_samplers_common, sd_unet, errors, rng, profiling
)
from modules.rng import slerp
from modules.sd_hijack import model_hijack
from modules.sd_samplers_common import (
    images_tensor_to_samples, decode_first_stage,
    approximation_indexes
)
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.paths as paths
import modules.face_restoration
import modules.images as images
import modules.styles
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion
from blendmodes.blend import blendLayers, BlendType

# Constants that should not be changed as they could break the model
OPT_CHANNEL_COUNT = 4  
OPT_F = 8

logger = logging.getLogger(__name__)

def setup_color_correction(image: Image.Image) -> np.ndarray:
    """
    Calibrate color correction for an image.
    
    Args:
        image: Input PIL image
        
    Returns:
        Color correction target as numpy array
    """
    logger.info("Calibrating color correction")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target

def apply_color_correction(
    correction: np.ndarray,
    original_image: Image.Image
) -> Image.Image:
    """
    Apply color correction to an image.
    
    Args:
        correction: Color correction target array
        original_image: Image to correct
        
    Returns:
        Color corrected PIL image
    """
    logger.info("Applying color correction")
    
    # Convert and match histograms
    image_array = np.asarray(original_image)
    corrected = cv2.cvtColor(
        exposure.match_histograms(
            cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB),
            correction,
            channel_axis=2
        ),
        cv2.COLOR_LAB2RGB
    ).astype("uint8")
    
    corrected_image = Image.fromarray(corrected)
    blended = blendLayers(corrected_image, original_image, BlendType.LUMINOSITY)
    
    return blended.convert('RGB')

def uncrop(
    image: Image.Image,
    dest_size: Tuple[int, int],
    paste_loc: Tuple[int, int, int, int]
) -> Image.Image:
    """
    Uncrop an image by pasting it into a larger canvas.
    
    Args:
        image: Image to uncrop
        dest_size: Final size (width, height)
        paste_loc: Location to paste (x, y, w, h)
        
    Returns:
        Uncropped image
    """
    x, y, w, h = paste_loc
    base_image = Image.new('RGBA', dest_size)
    resized = images.resize_image(1, image, w, h)
    base_image.paste(resized, (x, y))
    return base_image

def apply_overlay(
    image: Image.Image,
    paste_loc: Optional[Tuple[int, int, int, int]],
    overlay: Optional[Image.Image]
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply an overlay to an image.
    
    Args:
        image: Base image
        paste_loc: Paste location tuple
        overlay: Overlay image
        
    Returns:
        Tuple of (processed image, original copy)
    """
    if overlay is None:
        return image, image.copy()

    if paste_loc is not None:
        image = uncrop(image, (overlay.width, overlay.height), paste_loc)

    original = image.copy()
    
    # Apply overlay
    image = image.convert('RGBA')
    image.alpha_composite(overlay)
    image = image.convert('RGB')

    return image, original

def create_binary_mask(
    image: Image.Image,
    round_mask: bool = True
) -> Image.Image:
    """
    Create a binary mask from an RGBA image.
    
    Args:
        image: Input image
        round_mask: Whether to round mask values
        
    Returns:
        Binary mask as PIL image
    """
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        if round_mask:
            return image.split()[-1].convert("L").point(
                lambda x: 255 if x > 128 else 0
            )
        return image.split()[-1].convert("L")
    return image.convert('L')

def txt2img_image_conditioning(
    sd_model: Any,
    x: torch.Tensor,
    width: int,
    height: int
) -> torch.Tensor:
    """
    Get conditioning tensor for txt2img generation.
    
    Args:
        sd_model: Stable diffusion model
        x: Input tensor
        width: Image width 
        height: Image height
        
    Returns:
        Conditioning tensor
    """
    if sd_model.model.conditioning_key in {'hybrid', 'concat'}:
        # For inpainting models, create a masked image tensor
        image_conditioning = torch.ones(
            x.shape[0], 3, height, width,
            device=x.device
        ) * 0.5
        image_conditioning = images_tensor_to_samples(
            image_conditioning,
            approximation_indexes.get(opts.sd_vae_encode_method)
        )
        
        # Add full mask
        image_conditioning = torch.nn.functional.pad(
            image_conditioning,
            (0, 0, 0, 0, 1, 0),
            value=1.0
        )
        return image_conditioning.to(x.dtype)

    elif sd_model.model.conditioning_key == "crossattn-adm":
        # For UnCLIP models
        return x.new_zeros(
            x.shape[0],
            2*sd_model.noise_augmentor.time_embed.dim,
            dtype=x.dtype,
            device=x.device
        )
    
    else:
        if sd_model.is_sdxl_inpaint:
            # Handle SDXL inpainting
            image_conditioning = torch.ones(
                x.shape[0], 3, height, width,
                device=x.device
            ) * 0.5
            image_conditioning = images_tensor_to_samples(
                image_conditioning,
                approximation_indexes.get(opts.sd_vae_encode_method) 
            )
            image_conditioning = torch.nn.functional.pad(
                image_conditioning,
                (0, 0, 0, 0, 1, 0),
                value=1.0
            )
            return image_conditioning.to(x.dtype)

        # Default case - return dummy tensor
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)

# Main processing classes
@dataclass
class StableDiffusionProcessing:
    """Base class for stable diffusion processing."""
    
    sd_model: Any
    outpath_samples: Optional[str] = None
    outpath_grids: Optional[str] = None
    prompt: str = ""
    prompt_for_display: Optional[str] = None
    negative_prompt: str = ""
    styles: Optional[List[str]] = None
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    seed_enable_extras: bool = True
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    restore_faces: Optional[bool] = None
    tiling: Optional[bool] = None
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False
    extra_generation_params: Dict[str, Any] = field(default_factory=dict)
    overlay_images: Optional[List] = None
    eta: Optional[float] = None
    do_not_reload_embeddings: bool = False
    denoising_strength: Optional[float] = None
    ddim_discretize: Optional[str] = None
    s_min_uncond: Optional[float] = None
    s_churn: Optional[float] = None
    s_tmax: Optional[float] = None
    s_tmin: Optional[float] = None
    s_noise: Optional[float] = None
    override_settings: Dict[str, Any] = field(default_factory=dict)
    override_settings_restore_afterwards: bool = True
    sampler_index: Optional[int] = None
    refiner_checkpoint: Optional[str] = None
    refiner_switch_at: Optional[float] = None
    token_merging_ratio: float = 0
    token_merging_ratio_hr: float = 0
    disable_extra_networks: bool = False
    firstpass_image: Optional[Image.Image] = None

    # Internal state fields
    scripts_value: Optional[scripts.ScriptRunner] = field(default=None, init=False)
    script_args_value: Optional[List] = field(default=None, init=False)
    scripts_setup_complete: bool = field(default=False, init=False)
    cached_uc: List = field(default_factory=lambda: [None, None], init=False)
    cached_c: List = field(default_factory=lambda: [None, None], init=False)
    comments: Dict = field(default_factory=dict, init=False)
    sampler: Optional[sd_samplers_common.Sampler] = field(default=None, init=False)
    is_using_inpainting_conditioning: bool = field(default=False, init=False)
    paste_to: Optional[Tuple] = field(default=None, init=False)
    is_hr_pass: bool = field(default=False, init=False)
    c: Optional[Tuple] = field(default=None, init=False)
    uc: Optional[Tuple] = field(default=None, init=False)
    rng: Optional[rng.ImageRNG] = field(default=None, init=False)
    step_multiplier: int = field(default=1, init=False)
    color_corrections: Optional[List] = field(default=None, init=False)
    all_prompts: Optional[List] = field(default=None, init=False)
    all_negative_prompts: Optional[List] = field(default=None, init=False)
    all_seeds: Optional[List] = field(default=None, init=False)
    all_subseeds: Optional[List] = field(default=None, init=False)
    iteration: int = field(default=0, init=False)
    main_prompt: Optional[str] = field(default=None, init=False)
    main_negative_prompt: Optional[str] = field(default=None, init=False)
    prompts: Optional[List] = field(default=None, init=False)
    negative_prompts: Optional[List] = field(default=None, init=False)
    seeds: Optional[List] = field(default=None, init=False)
    subseeds: Optional[List] = field(default=None, init=False)
    extra_network_data: Optional[Dict] = field(default=None, init=False)
    user: Optional[str] = field(default=None, init=False)
    sd_model_name: Optional[str] = field(default=None, init=False)
    sd_model_hash: Optional[str] = field(default=None, init=False)
    sd_vae_name: Optional[str] = field(default=None, init=False)
    sd_vae_hash: Optional[str] = field(default=None, init=False)
    is_api: bool = field(default=False, init=False)

    def __post_init__(self):
        """Initialize after instance creation."""
        if self.sampler_index is not None:
            logger.warning(
                "sampler_index argument for StableDiffusionProcessing is deprecated; "
                "use sampler_name instead"
            )

        self.comments = {}
        self.styles = self.styles or []
        self.sampler_noise_scheduler_override = None
        self.extra_generation_params = self.extra_generation_params or {}
        self.override_settings = self.override_settings or {}
        self.script_args = self.script_args or {}
        self.refiner_checkpoint_info = None

        # Disable certain seed options if not enabled
        if not self.seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        # Set up caching
        self.cached_uc = StableDiffusionProcessing.cached_uc
        self.cached_c = StableDiffusionProcessing.cached_c

    def fill_fields_from_opts(self):
        """Fill fields from shared options."""
        self.s_min_uncond = (
            self.s_min_uncond if self.s_min_uncond is not None
            else opts.s_min_uncond
        )
        self.s_churn = (
            self.s_churn if self.s_churn is not None
            else opts.s_churn
        )
        self.s_tmin = (
            self.s_tmin if self.s_tmin is not None
            else opts.s_tmin
        )
        self.s_tmax = (
            self.s_tmax if self.s_tmax is not None
            else opts.s_tmax
        ) or float('inf')
        self.s_noise = (
            self.s_noise if self.s_noise is not None
            else opts.s_noise
        )

    @property
    def sd_model(self) -> Any:
        """Get the SD model."""
        return shared.sd_model

    @sd_model.setter 
    def sd_model(self, value: Any):
        """Setter for SD model (no-op)."""
        pass

    @property
    def scripts(self) -> Optional[scripts.ScriptRunner]:
        """Get scripts runner."""
        return self.scripts_value

    @scripts.setter
    def scripts(self, value: Optional[scripts.ScriptRunner]):
        """Set scripts runner and setup if needed."""
        self.scripts_value = value
        if (self.scripts_value and self.script_args_value
            and not self.scripts_setup_complete):
            self.setup_scripts()

    @property
    def script_args(self) -> Optional[List]:
        """Get script arguments."""
        return self.script_args_value 

    @script_args.setter
    def script_args(self, value: Optional[List]):
        """Set script arguments and setup if needed."""
        self.script_args_value = value
        if (self.scripts_value and self.script_args_value
            and not self.scripts_setup_complete):
            self.setup_scripts()

    def setup_scripts(self):
        """Set up scripts."""
        self.scripts_setup_complete = True
        self.scripts.setup_scrips(self, is_ui=not self.is_api)

    def comment(self, text: str):
        """Add a comment."""
        self.comments[text] = 1

    def txt2img_image_conditioning(
        self,
        x: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> torch.Tensor:
        """Get conditioning for txt2img generation."""
        self.is_using_inpainting_conditioning = (
            self.sd_model.model.conditioning_key in {'hybrid', 'concat'}
        )
        return txt2img_image_conditioning(
            self.sd_model,
            x,
            width or self.width,
            height or self.height
        )

    def depth2img_image_conditioning(
        self,
        source_image: torch.Tensor
    ) -> torch.Tensor:
        """Get conditioning for depth2img generation."""
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer({
            "jpg": rearrange(source_image[0], "c h w -> h w c")
        })
        midas_in = torch.from_numpy(
            transformed["midas_in"][None, ...]
        ).to(device=shared.device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)

        conditioning_image = images_tensor_to_samples(
            source_image * 0.5 + 0.5,
            approximation_indexes.get(opts.sd_vae_encode_method)
        )
        
        conditioning = torch.nn.functional.interpolate(
            self.sd_model.depth_model(midas_in),
            size=conditioning_image.shape[2:],
            mode="bicubic",
            align_corners=False
        )

        depth_min, depth_max = torch.aminmax(conditioning)
        conditioning = (
            2.0 * (conditioning - depth_min) / (depth_max - depth_min) - 1.0
        )
        return conditioning

    def edit_image_conditioning(
        self,
        source_image: torch.Tensor
    ) -> torch.Tensor:
        """Get conditioning for image editing."""
        return shared.sd_model.encode_first_stage(source_image).mode()

    def unclip_image_conditioning(
        self,
        source_image: torch.Tensor
    ) -> torch.Tensor:
        """Get conditioning for unCLIP models."""
        c_adm = self.sd_model.embedder(source_image)
        if self.sd_model.noise_augmentor is not None:
            noise_level = 0  # TODO: Allow other noise levels
            c_adm, noise_level_emb = self.sd_model.noise_augmentor(
                c_adm,
                noise_level=repeat(
                    torch.tensor([noise_level]).to(c_adm.device),
                    '1 -> b',
                    b=c_adm.shape[0]
                )
            )
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        return c_adm

    def inpainting_image_conditioning(
        self,
        source_image: torch.Tensor,
        latent_image: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        round_image_mask: bool = True
    ) -> torch.Tensor:
        """Get conditioning for inpainting."""
        self.is_using_inpainting_conditioning = True

        if image_mask is not None:
            if torch.is_tensor(image_mask):
                conditioning_mask = image_mask
            else:
                conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(
                    conditioning_mask[None, None]
                )

                if round_image_mask:
                    conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(
                1, 1, *source_image.shape[-2:]
            )

        # Create masked version of source image
        conditioning_mask = conditioning_mask.to(
            device=source_image.device,
            dtype=source_image.dtype
        )
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            getattr(
                self,
                "inpainting_mask_weight",
                shared.opts.inpainting_mask_weight
            )
        )

        # Encode masked image
        conditioning_image = self.sd_model.get_first_stage_encoding(
            self.sd_model.encode_first_stage(conditioning_image)
        )

        # Create concatenated conditioning tensor
        conditioning_mask = torch.nn.functional.interpolate(
            conditioning_mask,
            size=latent_image.shape[-2:]
        )
        conditioning_mask = conditioning_mask.expand(
            conditioning_image.shape[0], -1, -1, -1
        )
        image_conditioning = torch.cat(
            [conditioning_mask, conditioning_image],
            dim=1
        )
        return image_conditioning.to(shared.device).type(self.sd_model.dtype)

    def img2img_image_conditioning(
        self,
        source_image: torch.Tensor,
        latent_image: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        round_image_mask: bool = True
    ) -> torch.Tensor:
        """Get conditioning for img2img generation."""
        source_image = devices.cond_cast_float(source_image)

        if isinstance(self.sd_model, LatentDepth2ImageDiffusion):
            return self.depth2img_image_conditioning(source_image)

        if self.sd_model.cond_stage_key == "edit":
            return self.edit_image_conditioning(source_image)

        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            return self.inpainting_image_conditioning(
                source_image,
                latent_image,
                image_mask=image_mask,
                round_image_mask=round_image_mask
            )

        if self.sampler.conditioning_key == "crossattn-adm":
            return self.unclip_image_conditioning(source_image)

        if self.sampler.model_wrap.inner_model.is_sdxl_inpaint:
            return self.inpainting_image_conditioning(
                source_image,
                latent_image,
                image_mask=image_mask
            )

        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def init(
        self,
        all_prompts: List[str],
        all_seeds: List[int],
        all_subseeds: List[int]
    ):
        """Initialize processing."""
        pass

    def sample(
        self,
        conditioning: Any,
        unconditional_conditioning: Any,
        seeds: List[int],
        subseeds: List[int],
        subseed_strength: float,
        prompts: List[str]
    ) -> Any:
        """Generate samples."""
        raise NotImplementedError()

    def close(self):
        """Clean up resources."""
        self.sampler = None
        self.c = None
        self.uc = None
        if not opts.persistent_cond_cache:
            StableDiffusionProcessing.cached_c = [None, None]
            StableDiffusionProcessing.cached_uc = [None, None]

    def get_token_merging_ratio(self, for_hr: bool = False) -> float:
        """Get token merging ratio."""
        if for_hr:
            return (
                self.token_merging_ratio_hr
                or opts.token_merging_ratio_hr
                or self.token_merging_ratio
                or opts.token_merging_ratio
            )
        return (
            self.token_merging_ratio
            or opts.token_merging_ratio
        )

    def setup_prompts(self):
        """Set up prompts for generation."""
        if isinstance(self.prompt, list):
            self.all_prompts = self.prompt
        elif isinstance(self.negative_prompt, list):
            self.all_prompts = [self.prompt] * len(self.negative_prompt)
        else:
            self.all_prompts = self.batch_size * self.n_iter * [self.prompt]

        if isinstance(self.negative_prompt, list):
            self.all_negative_prompts = self.negative_prompt
        else:
            self.all_negative_prompts = [self.negative_prompt] * len(
                self.all_prompts
            )

        if len(self.all_prompts) != len(self.all_negative_prompts):
            raise RuntimeError(
                f"Got {len(self.all_prompts)} positive prompts and "
                f"{len(self.all_negative_prompts)} negative prompts"
            )

        self.all_prompts = [
            shared.prompt_styles.apply_styles_to_prompt(x, self.styles)
            for x in self.all_prompts
        ]
        self.all_negative_prompts = [
            shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles)
            for x in self.all_negative_prompts
        ]

        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]

    def cached_params(
        self,
        required_prompts: List[str],
        steps: int,
        extra_network_data: Dict,
        hires_steps: Optional[int] = None,
        use_old_scheduling: bool = False
    ) -> Tuple:
        """Get parameters for caching."""
        return (
            required_prompts,
            steps,
            hires_steps,
            use_old_scheduling,
            opts.CLIP_stop_at_last_layers,
            shared.sd_model.sd_checkpoint_info,
            extra_network_data,
            opts.sdxl_crop_left,
            opts.sdxl_crop_top,
            self.width,
            self.height,
            opts.fp8_storage,
            opts.cache_fp16_weight,
            opts.emphasis,
        )

    def get_conds_with_caching(
        self,
        function: Any,
        required_prompts: List[str],
        steps: int,
        caches: List,
        extra_network_data: Dict,
        hires_steps: Optional[int] = None
    ) -> Any:
        """Get conditioning with caching."""
        if shared.opts.use_old_scheduling:
            old_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(
                required_prompts, steps, hires_steps, False
            )
            new_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(
                required_prompts, steps, hires_steps, True
            )
            if old_schedules != new_schedules:
                self.extra_generation_params[
                    "Old prompt editing timelines"
                ] = True

        cached_params = self.cached_params(
            required_prompts,
            steps,
            extra_network_data,
            hires_steps,
            shared.opts.use_old_scheduling
        )

        for cache in caches:
            if cache[0] is not None and cached_params == cache[0]:
                return cache[1]

        cache = caches[0]

        with devices.autocast():
            cache[1] = function(
                shared.sd_model,
                required_prompts,
                steps,
                hires_steps,
                shared.opts.use_old_scheduling
            )

        cache[0] = cached_params
        return cache[1]

    def setup_conds(self):
        """Set up conditioning."""
        prompts = prompt_parser.SdConditioning(
            self.prompts,
            width=self.width,
            height=self.height
        )
        negative_prompts = prompt_parser.SdConditioning(
            self.negative_prompts,
            width=self.width,
            height=self.height,
            is_negative_prompt=True
        )

        sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
        total_steps = (
            sampler_config.total_steps(self.steps)
            if sampler_config else self.steps
        )
        self.step_multiplier = total_steps // self.steps
        self.firstpass_steps = total_steps

        self.uc = self.get_conds_with_caching(
            prompt_parser.get_learned_conditioning,
            negative_prompts,
            total_steps,
            [self.cached_uc],
            self.extra_network_data
        )
        self.c = self.get_conds_with_caching(
            prompt_parser.get_multicond_learned_conditioning,
            prompts,
            total_steps,
            [self.cached_c],
            self.extra_network_data
        )

    def get_conds(self) -> Tuple:
        """Get conditioning tensors."""
        return self.c, self.uc

    def parse_extra_network_prompts(self):
        """Parse prompts for extra networks."""
        self.prompts, self.extra_network_data = extra_networks.parse_prompts(
            self.prompts
        )

    def save_samples(self) -> bool:
        """Check if samples should be saved."""
        return (
            opts.samples_save
            and not self.do_not_save_samples
            and (
                opts.save_incomplete_images
                or not state.interrupted and not state.skipped
            )
        )

# Additional classes would follow here...