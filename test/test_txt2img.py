from dataclasses import dataclass
from typing import Any, Dict

import pytest
import requests
from requests import Response


@dataclass
class Txt2ImgRequest:
    batch_size: int = 1
    cfg_scale: int = 7
    denoising_strength: int = 0
    enable_hr: bool = False
    eta: int = 0
    firstphase_height: int = 0
    firstphase_width: int = 0
    height: int = 64
    n_iter: int = 1
    negative_prompt: str = ""
    prompt: str = "example prompt"
    restore_faces: bool = False
    s_churn: int = 0
    s_noise: int = 1
    s_tmax: int = 0
    s_tmin: int = 0
    sampler_index: str = "Euler a"
    seed: int = -1
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    steps: int = 3
    styles: list = None
    subseed: int = -1
    subseed_strength: int = 0
    tiling: bool = False
    width: int = 64

    def __post_init__(self) -> None:
        if self.styles is None:
            self.styles = []

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@pytest.fixture()
def url_txt2img(base_url: str) -> str:
    return f"{base_url}/sdapi/v1/txt2img"


@pytest.fixture()
def simple_txt2img_request() -> Dict[str, Any]:
    return Txt2ImgRequest().to_dict()


def test_txt2img_simple_performed(url_txt2img: str, simple_txt2img_request: Dict[str, Any]) -> None:
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200


def test_txt2img_with_negative_prompt_performed(url_txt2img: str, simple_txt2img_request: Dict[str, Any]) -> None:
    simple_txt2img_request["negative_prompt"] = "example negative prompt"
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200


def test_txt2img_with_complex_prompt_performed(url_txt2img: str, simple_txt2img_request: Dict[str, Any]) -> None:
    simple_txt2img_request["prompt"] = "((emphasis)), (emphasis1:1.1), [to:1], [from::2], [from:to:0.3], [alt|alt1]"
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200


def test_txt2img_not_square_image_performed(url_txt2img: str, simple_txt2img_request: Dict[str, Any]) -> None:
    simple_txt2img_request["height"] = 128
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200


def test_txt2img_with_hrfix_performed(url_txt2img: str, simple_txt2img_request: Dict[str, Any]) -> None:
    simple_txt2img_request["enable_hr"] = True
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200


def test_txt2img_with_tiling_performed(url_txt2img: str, simple_txt2img_request: Dict[str, Any]) -> None:
    simple_txt2img_request["tiling"] = True
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200


def test_txt2img_with_restore_faces_performed(url_txt2img: str, simple_txt2img_request: Dict[str, Any]) -> None:
    simple_txt2img_request["restore_faces"] = True
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200


@pytest.mark.parametrize("sampler", ["PLMS", "DDIM", "UniPC"])
def test_txt2img_with_vanilla_sampler_performed(
    url_txt2img: str, simple_txt2img_request: Dict[str, Any], sampler: str
) -> None:
    simple_txt2img_request["sampler_index"] = sampler
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200


def test_txt2img_multiple_batches_performed(url_txt2img: str, simple_txt2img_request: Dict[str, Any]) -> None:
    simple_txt2img_request["n_iter"] = 2
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200


def test_txt2img_batch_performed(url_txt2img: str, simple_txt2img_request: Dict[str, Any]) -> None:
    simple_txt2img_request["batch_size"] = 2
    response: Response = requests.post(url_txt2img, json=simple_txt2img_request)
    assert response.status_code == 200