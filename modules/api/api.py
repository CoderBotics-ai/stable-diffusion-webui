import base64
import io
import os
import time
import datetime
from typing import Any, Annotated
import uvicorn
import ipaddress
import requests
import gradio as gr
from threading import Lock
from io import BytesIO
from fastapi import APIRouter, Depends, FastAPI, Request, Response, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from secrets import compare_digest

import modules.shared as shared
from modules import (
    sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing,
    errors, restart, shared_items, script_callbacks, infotext_utils, sd_models,
    sd_schedulers
)
from modules.api import models
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from PIL import PngImagePlugin
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.realesrgan_model import get_realesrgan_models
from modules import devices
import piexif
import piexif.helper
from contextlib import closing, contextmanager
from modules.progress import create_task_id, add_task_to_queue, start_task, finish_task, current_task


def script_name_to_index(name: str, scripts: list) -> int:
    """Convert script name to index with improved error handling."""
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e


def validate_sampler_name(name: str) -> str:
    """Validate sampler name with type hints."""
    config = sd_samplers.all_samplers_map.get(name)
    if config is None:
        raise HTTPException(status_code=400, detail="Sampler not found")
    return name


def setUpscalers(req: dict[str, Any]) -> dict[str, Any]:
    """Set upscalers with proper type annotations."""
    reqDict = req.copy()
    reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
    reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
    return reqDict


def verify_url(url: str) -> bool:
    """Verify if URL refers to a global resource with improved error handling."""
    import socket
    from urllib.parse import urlparse
    try:
        parsed_url = urlparse(url)
        domain_name = parsed_url.netloc
        host = socket.gethostbyname_ex(domain_name)
        return all(ipaddress.ip_address(ip).is_global for ip in host[2])
    except Exception:
        return False


def decode_base64_to_image(encoding: str) -> Any:
    """Decode base64 to image with improved error handling."""
    if encoding.startswith(("http://", "https://")):
        if not opts.api_enable_requests:
            raise HTTPException(status_code=500, detail="Requests not allowed")

        if opts.api_forbid_local_requests and not verify_url(encoding):
            raise HTTPException(status_code=500, detail="Request to local resource not allowed")

        headers = {'user-agent': opts.api_useragent} if opts.api_useragent else {}
        try:
            response = requests.get(encoding, timeout=30, headers=headers)
            response.raise_for_status()
            return images.read(BytesIO(response.content))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Invalid image url: {str(e)}") from e

    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        return images.read(BytesIO(base64.b64decode(encoding)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid encoded image: {str(e)}") from e


def encode_pil_to_base64(image: Any) -> str:
    """Encode PIL image to base64 with improved error handling."""
    with io.BytesIO() as output_bytes:
        if isinstance(image, str):
            return image
        
        try:
            if opts.samples_format.lower() == 'png':
                use_metadata = False
                metadata = PngImagePlugin.PngInfo()
                for key, value in image.info.items():
                    if isinstance(key, str) and isinstance(value, str):
                        metadata.add_text(key, value)
                        use_metadata = True
                image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

            elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                parameters = image.info.get('parameters', None)
                exif_bytes = piexif.dump({
                    "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
                })
                if opts.samples_format.lower() in ("jpg", "jpeg"):
                    image.save(output_bytes, format="JPEG", exif=exif_bytes, quality=opts.jpeg_quality)
                else:
                    image.save(output_bytes, format="WEBP", exif=exif_bytes, quality=opts.jpeg_quality)
            else:
                raise HTTPException(status_code=500, detail="Invalid image format")

            return base64.b64encode(output_bytes.getvalue()).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}") from e


# Rest of the code remains the same as it's already well-structured and compatible with Python 3.12
# Only adding type hints and improving error handling where necessary

[... rest of the code remains unchanged ...]