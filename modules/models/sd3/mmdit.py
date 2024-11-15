import math
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from modules.models.sd3.other_impls import attention, Mlp


class PatchEmbed(nn.Module):
    """Converts a 2D image into patch embeddings."""
    
    def __init__(self, img_size: Optional[int] = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, flatten: bool = True, bias: bool = True,
                 strict_img_size: bool = True, dynamic_img_pad: bool = False,
                 dtype=None, device=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size) if img_size is not None else None
        self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size)) if self.img_size else None
        self.num_patches = self.grid_size[0] * self.grid_size[1] if self.grid_size else None

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                              bias=bias, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to project the input image into patch embeddings."""
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x


def modulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    """Modulates the input tensor with shift and scale."""
    shift = shift if shift is not None else torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False,
                             extra_tokens: int = 0, scaling_factor: Optional[float] = None,
                             offset: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generates 2D sine/cosine positional embeddings.
    
    Args:
        embed_dim: Dimension of the embeddings.
        grid_size: Size of the grid (height and width).
        cls_token: Whether to include a class token.
        extra_tokens: Number of extra tokens to prepend.
        scaling_factor: Factor to scale the grid.
        offset: Offset to apply to the grid.

    Returns:
        pos_embed: Positional embeddings of shape [grid_size*grid_size, embed_dim].
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    if scaling_factor is not None:
        grid /= scaling_factor
    if offset is not None:
        grid -= offset
        
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generates 2D sine/cosine positional embeddings from a grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generates 1D sine/cosine positional embeddings."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256,
                 dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Creates sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(dtype=t.dtype) if torch.is_floating_point(t) else embedding

    def forward(self, t: torch.Tensor, dtype) -> torch.Tensor:
        """Forward pass to embed timesteps."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        return self.mlp(t_freq)


class VectorEmbedder(nn.Module):
    """Embeds a flat vector of dimension input_dim."""

    def __init__(self, input_dim: int, hidden_size: int, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to embed the input vector."""
        return self.mlp(x)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class QkvLinear(torch.nn.Linear):
    """Linear layer for QKV projections."""
    pass

def split_qkv(qkv: torch.Tensor, head_dim: int) -> tuple:
    """Splits the QKV tensor into separate Q, K, and V tensors."""
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]

def optimized_attention(qkv: tuple, num_heads: int) -> torch.Tensor:
    """Applies optimized attention mechanism."""
    return attention(qkv[0], qkv[1], qkv[2], num_heads)

class SelfAttention(nn.Module):
    """Self-attention mechanism with various modes."""
    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_scale: Optional[float] = None, attn_mode: str = "xformers",
                 pre_only: bool = False, qk_norm: Optional[str] = None,
                 rmsnorm: bool = False, dtype=None, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = QkvLinear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = nn.Linear(dim, dim, dtype=dtype, device=device) if not pre_only else None
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()

    def pre_attention(self, x: torch.Tensor) -> tuple:
        """Prepares Q, K, V for attention."""
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Projects the output after attention."""
        assert not self.pre_only
        return self.proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the self-attention layer."""
        q, k, v = self.pre_attention(x)
        x = attention(q, k, v, self.num_heads)
        return self.post_attention(x)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype)) if self.learnable_scale else None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RMSNorm."""
        x = self._norm(x)
        return x * self.weight.to(device=x.device, dtype=x.dtype) if self.learnable_scale else x


class SwiGLUFeedForward(nn.Module):
    """Feedforward layer with SwiGLU activation."""
    
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float] = None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feedforward layer."""
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class DismantledBlock(nn.Module):
    """A DiT block with gated adaptive layer norm (adaLN) conditioning."""

    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_mode: str = "xformers", qkv_bias: bool = False,
                 pre_only: bool = False, rmsnorm: bool = False,
                 scale_mod_only: bool = False, swiglu: bool = False,
                 qk_norm: Optional[str] = None, dtype=None, device=None, **block_kwargs):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6) if rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=pre_only, qk_norm=qk_norm, rmsnorm=rmsnorm, dtype=dtype, device=device)
        self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6) if not pre_only and rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device) if not pre_only else None
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256) if swiglu else Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU(approximate="tanh"), dtype=dtype, device=device) if not pre_only else None
        
        self.scale_mod_only = scale_mod_only
        n_mods = 6 if not pre_only else 2 if not scale_mod_only else 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor) -> tuple:
        """Prepares Q, K, V for attention."""
        assert x is not None, "pre_attention called with None input"
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn: torch.Tensor, x: torch.Tensor, gate_msa: torch.Tensor, shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor) -> torch.Tensor:
        """Post-processes the attention output."""
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dismantled block."""
        assert not self.pre_only
        (q, k, v), intermediates = self.pre_attention(x, c)
        attn = attention(q, k, v, self.attn.num_heads)
        return self.post_attention(attn, *intermediates)


def block_mixing(context: torch.Tensor, x: torch.Tensor, context_block: DismantledBlock, x_block: DismantledBlock, c: torch.Tensor) -> tuple:
    """Mixes context and input blocks for attention."""
    assert context is not None, "block_mixing called with None context"
    context_qkv, context_intermediates = context_block.pre_attention(context, c)
    x_qkv, x_intermediates = x_block.pre_attention(x, c)

    q, k, v = [torch.cat((context_qkv[t], x_qkv[t]), dim=1) for t in range(3)]
    attn = attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = attn[:, :context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1]:]

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None
    x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


class JointBlock(nn.Module):
    """Wrapper for context and input blocks to facilitate mixed attention."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        qk_norm = kwargs.pop("qk_norm", None)
        self.context_block = DismantledBlock(*args, pre_only=pre_only, qk_norm=qk_norm, **kwargs)
        self.x_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through the joint block."""
        return block_mixing(*args, context_block=self.context_block, x_block=self.x_block, **kwargs)


class FinalLayer(nn.Module):
    """Final layer of the DiT model."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int,
                 total_out_channels: Optional[int] = None, dtype=None, device=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device) if total_out_channels is None else nn.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass through the final layer."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class MMDiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(self, input_size: int = 32, patch_size: int = 2, in_channels: int = 4,
                 depth: int = 28, mlp_ratio: float = 4.0, learn_sigma: bool = False,
                 adm_in_channels: Optional[int] = None, context_embedder_config: Optional[Dict] = None,
                 register_length: int = 0, attn_mode: str = "torch", rmsnorm: bool = False,
                 scale_mod_only: bool = False, swiglu: bool = False,
                 out_channels: Optional[int] = None, pos_embed_scaling_factor: Optional[float] = None,
                 pos_embed_offset: Optional[float] = None, pos_embed_max_size: Optional[int] = None,
                 num_patches=None, qk_norm: Optional[str] = None, qkv_bias: bool = True,
                 dtype=None, device=None):
        super().__init__()
        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size

        hidden_size = 64 * depth
        num_heads = depth

        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=self.pos_embed_max_size is None, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device)

        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = VectorEmbedder(adm_in_channels, hidden_size, dtype=dtype, device=device)

        self.context_embedder = nn.Identity()
        if context_embedder_config is not None and context_embedder_config["target"] == "torch.nn.Linear":
            self.context_embedder = nn.Linear(**context_embedder_config["params"], dtype=dtype, device=device)

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, hidden_size, dtype=dtype, device=device))

        if num_patches is not None:
            self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size, dtype=dtype, device=device))
        else:
            self.pos_embed = None

        self.joint_blocks = nn.ModuleList(
            [JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=i == depth - 1, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only, swiglu=swiglu, qk_norm=qk_norm, dtype=dtype, device=device) for i in range(depth)]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=dtype, device=device)

    def cropped_pos_embed(self, hw: tuple) -> torch.Tensor:
        """Crops the positional embedding based on the input size."""
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0]
        h, w = hw
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size and w <= self.pos_embed_max_size
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(self.pos_embed, "1 (h w) c -> 1 h w c", h=self.pos_embed_max_size, w=self.pos_embed_max_size)
        spatial_pos_embed = spatial_pos_embed[:, top: top + h, left: left + w, :]
        return rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")

    def unpatchify(self, x: torch.Tensor, hw: Optional[tuple] = None) -> torch.Tensor:
        """Converts the patched representation back to the original image shape."""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h //= p
            w //= p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward_core_with_concat(self, x: torch.Tensor, c_mod: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Core forward pass with concatenated context."""
        if self.register_length > 0:
            context = torch.cat((repeat(self.register, "1 ... -> b ...", b=x.shape[0]), context if context is not None else torch.Tensor([]).type_as(x)), 1)

        for block in self.joint_blocks:
            context, x = block(context, x, c=c_mod)

        return self.final_layer(x, c_mod)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the DiT model."""
        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(hw)
        c = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        if y is not None:
            y = self.y_embedder(y)  # (N, D)
            c += y  # (N, D)

        context = self.context_embedder(context)
        x = self.forward_core_with_concat(x, c, context)
        return self.unpatchify(x, hw=hw)  # (N, out_channels, H, W)