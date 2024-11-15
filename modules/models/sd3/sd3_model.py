import contextlib
import torch
import k_diffusion
from modules.models.sd3.sd3_impls import BaseModel, SDVAE, SD3LatentFormat
from modules.models.sd3.sd3_cond import SD3Cond
from modules import shared, devices


class SD3Denoiser(k_diffusion.external.DiscreteSchedule):
    """Denoiser class for the SD3 model, extending the discrete schedule."""
    
    def __init__(self, inner_model, sigmas):
        super().__init__(sigmas, quantize=shared.opts.enable_quantization)
        self.inner_model = inner_model

    def forward(self, input, sigma, **kwargs):
        """Applies the inner model to the input with the given sigma."""
        return self.inner_model.apply_model(input, sigma, **kwargs)


class SD3Inferencer(torch.nn.Module):
    """Inference class for the SD3 model, handling the model and its components."""
    
    def __init__(self, state_dict, shift=3, use_ema=False):
        super().__init__()
        self.shift = shift

        with torch.no_grad():
            self.model = BaseModel(shift=shift, state_dict=state_dict, prefix="model.diffusion_model.", device="cpu", dtype=devices.dtype)
            self.first_stage_model = SDVAE(device="cpu", dtype=devices.dtype_vae)
            self.first_stage_model.dtype = self.model.diffusion_model.dtype

        self.alphas_cumprod = 1 / (self.model.model_sampling.sigmas ** 2 + 1)
        self.text_encoders = SD3Cond()
        self.cond_stage_key = 'txt'
        self.parameterization = "eps"
        self.model.conditioning_key = "crossattn"
        self.latent_format = SD3LatentFormat()
        self.latent_channels = 16

    @property
    def cond_stage_model(self):
        """Returns the conditioning stage model."""
        return self.text_encoders

    def before_load_weights(self, state_dict):
        """Prepares the conditioning model before loading weights."""
        self.cond_stage_model.before_load_weights(state_dict)

    def ema_scope(self):
        """Context manager for EMA (Exponential Moving Average) operations."""
        return contextlib.nullcontext()

    def get_learned_conditioning(self, batch: list[str]):
        """Obtains learned conditioning from the text encoders."""
        return self.cond_stage_model(batch)

    def apply_model(self, x, t, cond):
        """Applies the model to the input with conditioning."""
        return self.model(x, t, c_crossattn=cond['crossattn'], y=cond['vector'])

    def decode_first_stage(self, latent):
        """Decodes the latent representation into an image."""
        latent = self.latent_format.process_out(latent)
        return self.first_stage_model.decode(latent)

    def encode_first_stage(self, image):
        """Encodes an image into its latent representation."""
        latent = self.first_stage_model.encode(image)
        return self.latent_format.process_in(latent)

    def get_first_stage_encoding(self, x):
        """Returns the first stage encoding."""
        return x

    def create_denoiser(self):
        """Creates a denoiser instance for the model."""
        return SD3Denoiser(self, self.model.model_sampling.sigmas)

    def medvram_fields(self):
        """Returns fields for memory-efficient training."""
        return [
            (self, 'first_stage_model'),
            (self, 'text_encoders'),
            (self, 'model'),
        ]

    def add_noise_to_latent(self, x, noise, amount):
        """Adds noise to the latent representation."""
        return x * (1 - amount) + noise * amount

    def fix_dimensions(self, width, height):
        """Adjusts dimensions to be multiples of 16."""
        return width // 16 * 16, height // 16 * 16

    def diffusers_weight_mapping(self):
        """Maps weights for the diffusion model."""
        for i in range(self.model.depth):
            yield f"transformer.transformer_blocks.{i}.attn.to_q", f"diffusion_model_joint_blocks_{i}_x_block_attn_qkv_q_proj"
            yield f"transformer.transformer_blocks.{i}.attn.to_k", f"diffusion_model_joint_blocks_{i}_x_block_attn_qkv_k_proj"
            yield f"transformer.transformer_blocks.{i}.attn.to_v", f"diffusion_model_joint_blocks_{i}_x_block_attn_qkv_v_proj"
            yield f"transformer.transformer_blocks.{i}.attn.to_out.0", f"diffusion_model_joint_blocks_{i}_x_block_attn_proj"
            yield f"transformer.transformer_blocks.{i}.attn.add_q_proj", f"diffusion_model_joint_blocks_{i}_context_block.attn_qkv_q_proj"
            yield f"transformer.transformer_blocks.{i}.attn.add_k_proj", f"diffusion_model_joint_blocks_{i}_context_block.attn_qkv_k_proj"
            yield f"transformer.transformer_blocks.{i}.attn.add_v_proj", f"diffusion_model_joint_blocks_{i}_context_block.attn_qkv_v_proj"
            yield f"transformer.transformer_blocks.{i}.attn.add_out_proj.0", f"diffusion_model_joint_blocks_{i}_context_block_attn_proj"