import os
from pathlib import Path
from typing import Optional, List

from modules.modelloader import load_file_from_url
from modules.upscaler import Upscaler, UpscalerData
from ldsr_model_arch import LDSR
from modules import shared, script_callbacks, errors
import sd_hijack_autoencoder  # noqa: F401
import sd_hijack_ddpm_v1  # noqa: F401


class UpscalerLDSR(Upscaler):
    """Upscaler implementation using LDSR model."""
    
    def __init__(self, user_path: str) -> None:
        """Initialize the LDSR upscaler.
        
        Args:
            user_path: Path to user directory for model storage
        """
        self.name = "LDSR"
        self.user_path = user_path
        self.model_url = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
        self.yaml_url = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
        super().__init__()
        scaler_data = UpscalerData("LDSR", None, self)
        self.scalers = [scaler_data]

    def load_model(self, path: str) -> LDSR:
        """Load the LDSR model from the specified path.
        
        Args:
            path: Path to the model directory
            
        Returns:
            Initialized LDSR model instance
        """
        model_dir = Path(self.model_path)
        yaml_path = model_dir / "project.yaml"
        old_model_path = model_dir / "model.pth"
        new_model_path = model_dir / "model.ckpt"

        local_model_paths = self.find_models(ext_filter=[".ckpt", ".safetensors"])
        local_ckpt_path = next(
            (local_model for local_model in local_model_paths 
             if local_model.endswith("model.ckpt")), None
        )
        local_safetensors_path = next(
            (local_model for local_model in local_model_paths 
             if local_model.endswith("model.safetensors")), None
        )
        local_yaml_path = next(
            (local_model for local_model in local_model_paths 
             if local_model.endswith("project.yaml")), None
        )

        if yaml_path.exists():
            statinfo = yaml_path.stat()
            if statinfo.st_size >= 10_485_760:  # 10 MB
                print("Removing invalid LDSR YAML file.")
                yaml_path.unlink()

        if old_model_path.exists():
            print("Renaming model from model.pth to model.ckpt")
            old_model_path.rename(new_model_path)

        model = (
            local_safetensors_path
            if local_safetensors_path is not None and Path(local_safetensors_path).exists()
            else local_ckpt_path or load_file_from_url(
                self.model_url, 
                model_dir=self.model_download_path, 
                file_name="model.ckpt"
            )
        )

        yaml = local_yaml_path or load_file_from_url(
            self.yaml_url, 
            model_dir=self.model_download_path, 
            file_name="project.yaml"
        )

        return LDSR(model, yaml)

    def do_upscale(self, img, path: str):
        """Perform upscaling using the LDSR model.
        
        Args:
            img: Input image to upscale
            path: Path to model directory
            
        Returns:
            Upscaled image or original image if upscaling fails
        """
        try:
            ldsr = self.load_model(path)
        except Exception as e:
            errors.report(f"Failed loading LDSR model {path}", exc_info=True)
            return img
            
        ddim_steps = shared.opts.ldsr_steps
        return ldsr.super_resolution(img, ddim_steps, self.scale)


def on_ui_settings():
    """Configure UI settings for LDSR upscaler."""
    import gradio as gr

    shared.opts.add_option(
        "ldsr_steps",
        shared.OptionInfo(
            100,
            "LDSR processing steps. Lower = faster",
            gr.Slider,
            {"minimum": 1, "maximum": 200, "step": 1},
            section=('upscaling', "Upscaling")
        )
    )
    shared.opts.add_option(
        "ldsr_cached",
        shared.OptionInfo(
            False,
            "Cache LDSR model in memory",
            gr.Checkbox,
            {"interactive": True},
            section=('upscaling', "Upscaling")
        )
    )


script_callbacks.on_ui_settings(on_ui_settings)