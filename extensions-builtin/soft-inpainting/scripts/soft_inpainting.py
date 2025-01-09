import numpy as np
import gradio as gr
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Dict
from modules.ui_components import InputAccordion
import modules.scripts as scripts
from modules.torch_utils import float64
from PIL import Image, ImageOps, ImageFilter
import torch
import modules.processing as proc
import modules.images as images

@dataclass
class SoftInpaintingSettings:
    mask_blend_power: float
    mask_blend_scale: float
    inpaint_detail_preservation: float
    composite_mask_influence: float
    composite_difference_threshold: float
    composite_difference_contrast: float

    def add_generation_params(self, dest: Dict[str, Any]) -> None:
        dest[enabled_gen_param_label] = True
        dest[gen_param_labels.mask_blend_power] = self.mask_blend_power
        dest[gen_param_labels.mask_blend_scale] = self.mask_blend_scale
        dest[gen_param_labels.inpaint_detail_preservation] = self.inpaint_detail_preservation
        dest[gen_param_labels.composite_mask_influence] = self.composite_mask_influence
        dest[gen_param_labels.composite_difference_threshold] = self.composite_difference_threshold
        dest[gen_param_labels.composite_difference_contrast] = self.composite_difference_contrast


def processing_uses_inpainting(p: Any) -> bool:
    """Check if the processing object uses inpainting by checking for mask attributes."""
    if getattr(p, "image_mask", None) is not None:
        return True

    if getattr(p, "mask", None) is not None:
        return True

    if getattr(p, "nmask", None) is not None:
        return True

    return False


def latent_blend(settings: SoftInpaintingSettings, 
                 a: torch.Tensor, 
                 b: torch.Tensor, 
                 t: torch.Tensor) -> torch.Tensor:
    """
    Interpolates two latent image representations according to the parameter t,
    where the interpolated vectors' magnitudes are also interpolated separately.
    The "detail_preservation" factor biases the magnitude interpolation towards
    the larger of the two magnitudes.

    Args:
        settings: Inpainting settings
        a: First latent tensor
        b: Second latent tensor
        t: Interpolation parameter tensor

    Returns:
        Interpolated latent tensor
    """
    if len(t.shape) == 3:
        t2 = t.unsqueeze(0)
        t3 = t[0].unsqueeze(0).unsqueeze(0)
    else:
        t2 = t
        t3 = t[:, 0][:, None]

    one_minus_t2 = 1 - t2
    one_minus_t3 = 1 - t3

    # Linearly interpolate the image vectors
    image_interp = a * one_minus_t2 + b * t2
    result_type = image_interp.dtype

    # Calculate and adjust magnitudes
    current_magnitude = torch.norm(image_interp, p=2, dim=1, keepdim=True).to(float64(image_interp)) + 0.00001
    a_magnitude = torch.norm(a, p=2, dim=1, keepdim=True).to(float64(a)).pow_(settings.inpaint_detail_preservation) * one_minus_t3
    b_magnitude = torch.norm(b, p=2, dim=1, keepdim=True).to(float64(b)).pow_(settings.inpaint_detail_preservation) * t3
    desired_magnitude = (a_magnitude + b_magnitude).pow_(1 / settings.inpaint_detail_preservation)

    # Adjust the interpolated image vectors' magnitudes
    image_interp_scaling_factor = (desired_magnitude / current_magnitude).to(result_type)
    return image_interp * image_interp_scaling_factor


def get_modified_nmask(settings: SoftInpaintingSettings, 
                       nmask: torch.Tensor, 
                       sigma: float) -> torch.Tensor:
    """
    Converts a negative mask representing the transparency of the original latent vectors
    to a mask scaled according to the denoising strength.

    Args:
        settings: Inpainting settings
        nmask: Negative mask tensor
        sigma: Denoising strength

    Returns:
        Modified negative mask tensor
    """
    return torch.pow(nmask, (sigma ** settings.mask_blend_power) * settings.mask_blend_scale)


def apply_adaptive_masks(
        settings: SoftInpaintingSettings,
        nmask: torch.Tensor,
        latent_orig: torch.Tensor,
        latent_processed: torch.Tensor,
        overlay_images: List[Image.Image],
        width: int,
        height: int,
        paste_to: Optional[Tuple[int, int]] = None) -> List[Image.Image]:
    """
    Applies adaptive masks to overlay images based on latent space differences.

    Args:
        settings: Inpainting settings
        nmask: Negative mask tensor
        latent_orig: Original latent tensor
        latent_processed: Processed latent tensor
        overlay_images: List of overlay images
        width: Target width
        height: Target height
        paste_to: Optional paste coordinates

    Returns:
        List of masked overlay images
    """
    if len(nmask.shape) == 3:
        latent_mask = nmask[0].float()
    else:
        latent_mask = nmask[:, 0].float()

    mask_scalar = 1 - (torch.clamp(latent_mask, min=0, max=1) ** (settings.mask_blend_scale / 2))
    mask_scalar = (0.5 * (1 - settings.composite_mask_influence) +
                  mask_scalar * settings.composite_mask_influence)
    mask_scalar = mask_scalar / (1.00001 - mask_scalar)
    mask_scalar = mask_scalar.cpu().numpy()

    latent_distance = torch.norm(latent_processed - latent_orig, p=2, dim=1)
    kernel, kernel_center = get_gaussian_kernel(stddev_radius=1.5, max_radius=2)
    masks_for_overlay = []

    for i, (distance_map, overlay_image) in enumerate(zip(latent_distance, overlay_images)):
        converted_mask = distance_map.float().cpu().numpy()
        converted_mask = weighted_histogram_filter(converted_mask, kernel, kernel_center,
                                                  percentile_min=0.9, percentile_max=1, min_width=1)
        converted_mask = weighted_histogram_filter(converted_mask, kernel, kernel_center,
                                                  percentile_min=0.25, percentile_max=0.75, min_width=1)

        half_weighted_distance = settings.composite_difference_threshold * (
            mask_scalar[i] if len(mask_scalar.shape) == 3 and mask_scalar.shape[0] > i else mask_scalar[0]
        )

        converted_mask = converted_mask / half_weighted_distance
        converted_mask = 1 / (1 + converted_mask ** settings.composite_difference_contrast)
        converted_mask = smootherstep(converted_mask)
        converted_mask = 1 - converted_mask
        converted_mask = (255. * converted_mask).astype(np.uint8)
        converted_mask = Image.fromarray(converted_mask)
        converted_mask = images.resize_image(2, converted_mask, width, height)
        converted_mask = proc.create_binary_mask(converted_mask, round=False)
        converted_mask = converted_mask.filter(ImageFilter.GaussianBlur(radius=4))

        if paste_to is not None:
            converted_mask = proc.uncrop(converted_mask,
                                        (overlay_image.width, overlay_image.height),
                                        paste_to)

        masks_for_overlay.append(converted_mask)
        image_masked = Image.new('RGBa', (overlay_image.width, overlay_image.height))
        image_masked.paste(overlay_image.convert("RGBA").convert("RGBa"),
                          mask=ImageOps.invert(converted_mask.convert('L')))
        overlay_images[i] = image_masked.convert('RGBA')

    return masks_for_overlay

[... rest of the file remains the same with similar type hint and docstring improvements ...]