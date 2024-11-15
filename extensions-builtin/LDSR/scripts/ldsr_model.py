import os
from modules.modelloader import load_file_from_url
from modules.upscaler import Upscaler, UpscalerData
from ldsr_model_arch import LDSR
from modules import shared, script_callbacks, errors
import sd_hijack_autoencoder  # noqa: F401
import sd_hijack_ddpm_v1  # noqa: F401


class UpscalerLDSR(Upscaler):
    """LDSR Upscaler class for loading and applying the LDSR model for image super-resolution."""

    def __init__(self, user_path: str):
        """Initialize the UpscalerLDSR with the user path and model URLs."""
        super().__init__()
        self.name = "LDSR"
        self.user_path = user_path
        self.model_url = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
        self.yaml_url = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
        self.scalers = [UpscalerData("LDSR", None, self)]

    def load_model(self, path: str) -> LDSR:
        """Load the LDSR model from the specified path or download it if not available."""
        # Define paths for model files
        yaml_path = os.path.join(self.model_path, "project.yaml")
        old_model_path = os.path.join(self.model_path, "model.pth")
        new_model_path = os.path.join(self.model_path, "model.ckpt")

        # Check for existing model files
        local_model_paths = self.find_models(ext_filter=[".ckpt", ".safetensors"])
        local_ckpt_path = next((m for m in local_model_paths if m.endswith("model.ckpt")), None)
        local_safetensors_path = next((m for m in local_model_paths if m.endswith("model.safetensors")), None)
        local_yaml_path = next((m for m in local_model_paths if m.endswith("project.yaml")), None)

        # Remove invalid YAML file if too large
        if os.path.exists(yaml_path) and os.stat(yaml_path).st_size >= 10485760:
            print("Removing invalid LDSR YAML file.")
            os.remove(yaml_path)

        # Rename old model file if it exists
        if os.path.exists(old_model_path):
            print("Renaming model from model.pth to model.ckpt")
            os.rename(old_model_path, new_model_path)

        # Load model from local or download if not found
        model = (local_safetensors_path if local_safetensors_path and os.path.exists(local_safetensors_path)
                  else local_ckpt_path or load_file_from_url(self.model_url, model_dir=self.model_download_path, file_name="model.ckpt"))

        yaml = local_yaml_path or load_file_from_url(self.yaml_url, model_dir=self.model_download_path, file_name="project.yaml")

        return LDSR(model, yaml)

    def do_upscale(self, img, path: str):
        """Perform super-resolution on the given image using the LDSR model."""
        try:
            ldsr = self.load_model(path)
        except Exception as e:
            errors.report(f"Failed loading LDSR model {path}", exc_info=e)
            return img

        ddim_steps = shared.opts.ldsr_steps
        return ldsr.super_resolution(img, ddim_steps, self.scale)


def on_ui_settings():
    """Configure UI settings for the LDSR upscaling options."""
    import gradio as gr

    shared.opts.add_option("ldsr_steps", shared.OptionInfo(100, "LDSR processing steps. Lower = faster", gr.Slider, {"minimum": 1, "maximum": 200, "step": 1}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("ldsr_cached", shared.OptionInfo(False, "Cache LDSR model in memory", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")))


script_callbacks.on_ui_settings(on_ui_settings)