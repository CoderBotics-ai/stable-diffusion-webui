from typing import Any
import requests
from requests import Response


def test_simple_upscaling_performed(base_url: str, img2img_basic_image_base64: str) -> None:
    """
    Test the upscaling functionality of the API.
    
    Args:
        base_url: The base URL of the API
        img2img_basic_image_base64: Base64 encoded image data
    """
    payload: dict[str, Any] = {
        "resize_mode": 0,
        "show_extras_results": True,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscaling_resize": 2,
        "upscaling_resize_w": 128,
        "upscaling_resize_h": 128,
        "upscaling_crop": True,
        "upscaler_1": "Lanczos",
        "upscaler_2": "None",
        "extras_upscaler_2_visibility": 0,
        "image": img2img_basic_image_base64,
    }
    response: Response = requests.post(
        f"{base_url}/sdapi/v1/extra-single-image",
        json=payload
    )
    assert response.status_code == 200


def test_png_info_performed(base_url: str, img2img_basic_image_base64: str) -> None:
    """
    Test the PNG information extraction functionality of the API.
    
    Args:
        base_url: The base URL of the API
        img2img_basic_image_base64: Base64 encoded image data
    """
    payload: dict[str, str] = {
        "image": img2img_basic_image_base64,
    }
    response: Response = requests.post(
        f"{base_url}/sdapi/v1/extra-single-image",
        json=payload
    )
    assert response.status_code == 200


def test_interrogate_performed(base_url: str, img2img_basic_image_base64: str) -> None:
    """
    Test the image interrogation functionality of the API using the CLIP model.
    
    Args:
        base_url: The base URL of the API
        img2img_basic_image_base64: Base64 encoded image data
    """
    payload: dict[str, str] = {
        "image": img2img_basic_image_base64,
        "model": "clip",
    }
    response: Response = requests.post(
        f"{base_url}/sdapi/v1/extra-single-image",
        json=payload
    )
    assert response.status_code == 200