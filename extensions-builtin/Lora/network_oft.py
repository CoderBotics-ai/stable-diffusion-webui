from typing import Optional, Type
import torch
import torch.nn as nn
from torch import Tensor
import network
from einops import rearrange


class ModuleTypeOFT(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights) -> Optional['NetworkModuleOFT']:
        """Create a NetworkModuleOFT if the required weights are present."""
        match weights.w:
            case {'oft_blocks': _} | {'oft_diag': _}:
                return NetworkModuleOFT(net, weights)
            case _:
                return None


class NetworkModuleOFT(network.NetworkModule):
    """Network module implementation for Orthogonal Function Transformations (OFT)."""
    
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        super().__init__(net, weights)

        self.lin_module: Optional[nn.Module] = None
        self.org_module: list[torch.Module] = [self.sd_module]

        self.scale: float = 1.0
        self.is_R: bool = False
        self.is_boft: bool = False

        # Initialize OFT parameters based on weight type
        match weights.w:
            case {'oft_blocks': blocks, **rest}:
                # kohya-ss/New LyCORIS OFT/BOFT
                self.oft_blocks: Tensor = blocks
                self.alpha: Optional[Tensor] = rest.get('alpha')
                self.dim: int = self.oft_blocks.shape[0]
            case {'oft_diag': diag}:
                # Old LyCORIS OFT
                self.is_R = True
                self.oft_blocks = diag
                self.alpha = None
                self.dim = self.oft_blocks.shape[1]
            case _:
                raise ValueError("Invalid weights configuration for OFT")

        # Determine module type and output dimension
        module_type: Type[nn.Module] = type(self.sd_module)
        self.out_dim: int = self._get_output_dimension(module_type)

        # Handle BOFT specific initialization
        self.is_boft = self.oft_blocks.dim() == 4
        self.rescale: Optional[Tensor] = weights.w.get('rescale')
        if self.rescale is not None and not isinstance(self.sd_module, nn.MultiheadAttention):
            self.rescale = self.rescale.reshape(-1, *[1] * (self.org_module[0].weight.dim() - 1))

        # Set block parameters
        self._initialize_block_parameters()

    def _get_output_dimension(self, module_type: Type[nn.Module]) -> int:
        """Determine the output dimension based on module type."""
        match module_type:
            case nn.Linear | nn.modules.linear.NonDynamicallyQuantizableLinear:
                return self.sd_module.out_features
            case nn.Conv2d:
                return self.sd_module.out_channels
            case nn.MultiheadAttention:
                return self.sd_module.embed_dim
            case _:
                raise ValueError(f"Unsupported module type: {module_type}")

    def _initialize_block_parameters(self) -> None:
        """Initialize block-related parameters based on OFT type."""
        self.num_blocks = self.dim
        self.block_size = self.out_dim // self.dim
        self.constraint = (0 if self.alpha is None else self.alpha) * self.out_dim

        if self.is_R:
            self.constraint = None
            self.block_size = self.dim
            self.num_blocks = self.out_dim // self.dim
        elif self.is_boft:
            self.boft_m = self.oft_blocks.shape[0]
            self.num_blocks = self.oft_blocks.shape[1]
            self.block_size = self.oft_blocks.shape[2]
            self.boft_b = self.block_size

    def calc_updown(self, orig_weight: Tensor) -> Tensor:
        """Calculate the weight update for OFT transformation."""
        oft_blocks = self.oft_blocks.to(orig_weight.device)
        eye = torch.eye(self.block_size, device=oft_blocks.device)

        if not self.is_R:
            block_Q = oft_blocks - oft_blocks.transpose(-1, -2)
            if self.constraint != 0:
                norm_Q = torch.norm(block_Q.flatten())
                new_norm_Q = torch.clamp(norm_Q, max=self.constraint.to(oft_blocks.device))
                block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
            oft_blocks = torch.matmul(eye + block_Q, (eye - block_Q).float().inverse())

        R = oft_blocks.to(orig_weight.device)

        if not self.is_boft:
            merged_weight = rearrange(orig_weight, '(k n) ... -> k n ...', k=self.num_blocks, n=self.block_size)
            merged_weight = torch.einsum(
                'k n m, k n ... -> k m ...',
                R,
                merged_weight
            )
            merged_weight = rearrange(merged_weight, 'k m ... -> (k m) ...')
        else:
            scale = 1.0
            m = self.boft_m
            b = self.boft_b
            r_b = b // 2
            inp = orig_weight
            for i in range(m):
                bi = R[i]
                if i == 0:
                    bi = bi * scale + (1 - scale) * eye
                inp = rearrange(inp, "(c g k) ... -> (c k g) ...", g=2, k=2**i * r_b)
                inp = rearrange(inp, "(d b) ... -> d b ...", b=b)
                inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
                inp = rearrange(inp, "d b ... -> (d b) ...")
                inp = rearrange(inp, "(c k g) ... -> (c g k) ...", g=2, k=2**i * r_b)
            merged_weight = inp

        if self.rescale is not None:
            merged_weight = self.rescale.to(merged_weight) * merged_weight

        updown = merged_weight.to(orig_weight.device) - orig_weight.to(merged_weight.dtype)
        output_shape = orig_weight.shape
        return self.finalize_updown(updown, orig_weight, output_shape)