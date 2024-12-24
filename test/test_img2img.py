from dataclasses import dataclass
from typing import Any, Optional

import pytest
import requests
from requests import Response


@dataclass
class Img2ImgRequest:
    """Data structure for img2img request parameters."""
    batch_size: int
    cfg_scale: float
    denoising_strength: float
    eta: int
    height: int
    include_init_images: bool
    init_images: list[str]
    inpaint_full_res: bool
    inpaint_full_res_padding: int
    inpainting_fill: int
    inpainting_mask_invert: bool
    mask: Optional[str]
    mask_blur: int
    n_iter: int
    negative_prompt: str
    override_settings: dict[str, Any]
    prompt: str
    resize_mode: int
    restore_faces: bool
    s_churn: int
    s_noise: int
    s_tmax: int
    s_tmin: int
    sampler_index: str
    seed: int
    seed_resize_from_h: int
    seed_resize_from_w: int
    steps: int
    styles: list[str]
    subseed: int
    subseed_strength: int
    tiling: bool
    width: int
    script_name: Optional[str] = None
    script_args: Optional[list[Any]] = None


@pytest.fixture()
def url_img2img(base_url: str) -> str:
    """Generate the img2img API endpoint URL."""
    return f"{base_url}/sdapi/v1/img2img"


@pytest.fixture()
def simple_img2img_request(img2img_basic_image_base64: str) -> dict[str, Any]:
    """Create a basic img2img request configuration."""
    return {
        "batch_size": 1,
        "cfg_scale": 7,
        "denoising_strength": 0.75,
        "eta": 0,
        "height": 64,
        "include_init_images": False,
        "init_images": [img2img_basic_image_base64],
        "inpaint_full_res": False,
        "inpaint_full_res_padding": 0,
        "inpainting_fill": 0,
        "inpainting_mask_invert": False,
        "mask": None,
        "mask_blur": 4,
        "n_iter": 1,
        "negative_prompt": "",
        "override_settings": {},
        "prompt": "example prompt",
        "resize_mode": 0,
        "restore_faces": False,
        "s_churn": 0,
        "s_noise": 1,
        "s_tmax": 0,
        "s_tmin": 0,
        "sampler_index": "Euler a",
        "seed": -1,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "steps": 3,
        "styles": [],
        "subseed": -1,
        "subseed_strength": 0,
        "tiling": False,
        "width": 64,
    }


def test_img2img_simple_performed(url_img2img: str, simple_img2img_request: dict[str, Any]) -> None:
    """Test basic img2img conversion."""
    response: Response = requests.post(url_img2img, json=simple_img2img_request)
    assert response.status_code == 200


def test_inpainting_masked_performed(
    url_img2img: str,
    simple_img2img_request: dict[str, Any],
    mask_basic_image_base64: str
) -> None:
    """Test img2img with mask applied."""
    simple_img2img_request["mask"] = mask_basic_image_base64
    response: Response = requests.post(url_img2img, json=simple_img2img_request)
    assert response.status_code == 200


def test_inpainting_with_inverted_masked_performed(
    url_img2img: str,
    simple_img2img_request: dict[str, Any],
    mask_basic_image_base64: str
) -> None:
    """Test img2img with inverted mask applied."""
    simple_img2img_request["mask"] = mask_basic_image_base64
    simple_img2img_request["inpainting_mask_invert"] = True
    response: Response = requests.post(url_img2img, json=simple_img2img_request)
    assert response.status_code == 200


def test_img2img_sd_upscale_performed(url_img2img: str, simple_img2img_request: dict[str, Any]) -> None:
    """Test img2img with SD upscale script."""
    simple_img2img_request["script_name"] = "sd upscale"
    simple_img2img_request["script_args"] = ["", 8, "Lanczos", 2.0]
    response: Response = requests.post(url_img2img, json=simple_img2img_request)
    assert response.status_code == 200