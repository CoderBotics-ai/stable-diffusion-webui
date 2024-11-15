### Implementations of the SD3 Core Diffusion Model and VAE

import torch
import math
import einops
from modules.models.sd3.mmdit import MMDiT
from PIL import Image


#################################################################################################
### MMDiT Model Wrapping
#################################################################################################

class ModelSamplingDiscreteFlow(torch.nn.Module):
    """Helper for sampler scheduling (i.e., timestep/sigma calculations) for Discrete Flow models."""
    
    def __init__(self, shift: float = 1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma: torch.Tensor, model_output: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor, max_denoise: bool = False) -> torch.Tensor:
        return sigma * noise + (1.0 - sigma) * latent_image


class BaseModel(torch.nn.Module):
    """Wrapper around the core MM-DiT model."""
    
    def __init__(self, shift: float = 1.0, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32, state_dict: Optional[dict] = None, prefix: str = ""):
        super().__init__()
        patch_size = state_dict[f"{prefix}x_embedder.proj.weight"].shape[2]
        depth = state_dict[f"{prefix}x_embedder.proj.weight"].shape[0] // 64
        num_patches = state_dict[f"{prefix}pos_embed"].shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = state_dict[f"{prefix}y_embedder.mlp.0.weight"].shape[1]
        context_shape = state_dict[f"{prefix}context_embedder.weight"].shape
        
        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0]
            }
        }
        
        self.diffusion_model = MMDiT(
            input_size=None,
            pos_embed_scaling_factor=None,
            pos_embed_offset=None,
            pos_embed_max_size=pos_embed_max_size,
            patch_size=patch_size,
            in_channels=16,
            depth=depth,
            num_patches=num_patches,
            adm_in_channels=adm_in_channels,
            context_embedder_config=context_embedder_config,
            device=device,
            dtype=dtype
        )
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)
        self.depth = depth

    def apply_model(self, x: torch.Tensor, sigma: torch.Tensor, c_crossattn: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        model_output = self.diffusion_model(x.to(dtype), timestep, context=c_crossattn.to(dtype), y=y.to(dtype)).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.apply_model(*args, **kwargs)

    def get_dtype(self) -> torch.dtype:
        return self.diffusion_model.dtype


class CFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs."""
    
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, cond: dict, uncond: dict, cond_scale: float) -> torch.Tensor:
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]])
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled


class SD3LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift."""
    
    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent / self.scale_factor) + self.shift_factor

    def decode_latent_to_preview(self, x0: torch.Tensor) -> Image:
        """Quick RGB approximate preview of sd3 latents."""
        factors = torch.tensor([
            [-0.0645,  0.0177,  0.1052], [ 0.0028,  0.0312,  0.0650],
            [ 0.1848,  0.0762,  0.0360], [ 0.0944,  0.0360,  0.0889],
            [ 0.0897,  0.0506, -0.0364], [-0.0020,  0.1203,  0.0284],
            [ 0.0855,  0.0118,  0.0283], [-0.0539,  0.0658,  0.1047],
            [-0.0057,  0.0116,  0.0700], [-0.0412,  0.0281, -0.0039],
            [ 0.1106,  0.1171,  0.1220], [-0.0248,  0.0682, -0.0481],
            [ 0.0815,  0.0846,  0.1207], [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456], [-0.1418, -0.1457, -0.1259]
        ], device="cpu")
        latent_image = x0[0].permute(1, 2, 0).cpu() @ factors

        latents_ubyte = (((latent_image + 1) / 2)
                            .clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            .byte()).cpu()

        return Image.fromarray(latents_ubyte.numpy())


#################################################################################################
### K-Diffusion Sampling
#################################################################################################

def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_euler(model: BaseModel, x: torch.Tensor, sigmas: torch.Tensor, extra_args: Optional[dict] = None) -> torch.Tensor:
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


#################################################################################################
### VAE
#################################################################################################

def Normalize(in_channels: int, num_groups: int = 32, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.nn.GroupNorm:
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)


class ResnetBlock(torch.nn.Module):
    def __init__(self, *, in_channels: int, out_channels: Optional[int] = None, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device) if self.in_channels != self.out_channels else None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(x)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden