from typing import Optional, Type
import torch
import torch.nn as nn
from torch import Tensor
import network
from einops import rearrange


class ModuleTypeOFT(network.ModuleType):
    def create_module(
        self, 
        net: network.Network, 
        weights: network.NetworkWeights
    ) -> Optional['NetworkModuleOFT']:
        """Create an OFT network module if the weights contain necessary components."""
        match weights.w:
            case {'oft_blocks': _} | {'oft_diag': _}:
                return NetworkModuleOFT(net, weights)
            case _:
                return None


class NetworkModuleOFT(network.NetworkModule):
    """Network module implementing Orthogonal Function Transformations (OFT).
    
    Supports both kohya-ss' implementation of COFT and KohakuBlueleaf's implementation of OFT/COFT.
    """
    
    def __init__(self, net: network.Network, weights: network.NetworkWeights) -> None:
        super().__init__(net, weights)

        self.lin_module: Optional[nn.Module] = None
        self.org_module: list[torch.Module] = [self.sd_module]
        
        # Initialize attributes with type hints
        self.scale: float = 1.0
        self.is_R: bool = False
        self.is_boft: bool = False
        self.oft_blocks: Tensor
        self.alpha: Optional[Tensor] = None
        self.dim: int
        self.out_dim: int
        self.rescale: Optional[Tensor] = None
        self.num_blocks: int
        self.block_size: int
        self.constraint: Optional[Tensor] = None
        
        # Initialize based on weight type
        if "oft_blocks" in weights.w:
            self._init_new_oft(weights)
        elif "oft_diag" in weights.w:
            self._init_old_oft(weights)
        
        self._setup_module_dimensions()
        self._setup_block_parameters()

    def _init_new_oft(self, weights: network.NetworkWeights) -> None:
        """Initialize for new OFT/BOFT implementation."""
        self.oft_blocks = weights.w["oft_blocks"]
        self.alpha = weights.w.get("alpha", None)
        self.dim = self.oft_blocks.shape[0]

    def _init_old_oft(self, weights: network.NetworkWeights) -> None:
        """Initialize for old OFT implementation."""
        self.is_R = True
        self.oft_blocks = weights.w["oft_diag"]
        self.dim = self.oft_blocks.shape[1]

    def _setup_module_dimensions(self) -> None:
        """Set up module dimensions based on the SD module type."""
        match self.sd_module:
            case nn.Linear() | nn.modules.linear.NonDynamicallyQuantizableLinear():
                self.out_dim = self.sd_module.out_features
            case nn.Conv2d():
                self.out_dim = self.sd_module.out_channels
            case nn.MultiheadAttention():
                self.out_dim = self.sd_module.embed_dim
            case _:
                raise ValueError(f"Unsupported module type: {type(self.sd_module)}")

    def _setup_block_parameters(self) -> None:
        """Set up block-related parameters and constraints."""
        if self.oft_blocks.dim() == 4:
            self.is_boft = True
            
        self.rescale = self.weights.w.get('rescale', None)
        if self.rescale is not None and not isinstance(self.sd_module, nn.MultiheadAttention):
            self.rescale = self.rescale.reshape(-1, *[1]*(self.org_module[0].weight.dim() - 1))

        if self.is_R:
            self._setup_r_parameters()
        elif self.is_boft:
            self._setup_boft_parameters()
        else:
            self._setup_standard_parameters()

    def _setup_r_parameters(self) -> None:
        """Set up parameters for R-type OFT."""
        self.constraint = None
        self.block_size = self.dim
        self.num_blocks = self.out_dim // self.dim

    def _setup_boft_parameters(self) -> None:
        """Set up parameters for BOFT."""
        self.boft_m = self.oft_blocks.shape[0]
        self.num_blocks = self.oft_blocks.shape[1]
        self.block_size = self.oft_blocks.shape[2]
        self.boft_b = self.block_size

    def _setup_standard_parameters(self) -> None:
        """Set up parameters for standard OFT."""
        self.num_blocks = self.dim
        self.block_size = self.out_dim // self.dim
        self.constraint = (0 if self.alpha is None else self.alpha) * self.out_dim

    def calc_updown(self, orig_weight: Tensor) -> Tensor:
        """Calculate the update-downdate tensor for the weights."""
        oft_blocks = self.oft_blocks.to(orig_weight.device)
        eye = torch.eye(self.block_size, device=oft_blocks.device)

        if not self.is_R:
            oft_blocks = self._process_non_r_blocks(oft_blocks, eye)

        R = oft_blocks.to(orig_weight.device)

        if not self.is_boft:
            merged_weight = self._process_standard_weight(orig_weight, R)
        else:
            merged_weight = self._process_boft_weight(orig_weight, R, eye)

        if self.rescale is not None:
            merged_weight = self.rescale.to(merged_weight) * merged_weight

        updown = merged_weight.to(orig_weight.device) - orig_weight.to(merged_weight.dtype)
        return self.finalize_updown(updown, orig_weight, orig_weight.shape)

    def _process_non_r_blocks(self, oft_blocks: Tensor, eye: Tensor) -> Tensor:
        """Process non-R type blocks to ensure skew-symmetric orthogonal matrix."""
        block_Q = oft_blocks - oft_blocks.transpose(-1, -2)
        if self.constraint != 0:
            norm_Q = torch.norm(block_Q.flatten())
            new_norm_Q = torch.clamp(norm_Q, max=self.constraint.to(oft_blocks.device))
            block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
        return torch.matmul(eye + block_Q, (eye - block_Q).float().inverse())

    def _process_standard_weight(self, orig_weight: Tensor, R: Tensor) -> Tensor:
        """Process weights for standard OFT."""
        merged_weight = rearrange(orig_weight, '(k n) ... -> k n ...', k=self.num_blocks, n=self.block_size)
        merged_weight = torch.einsum('k n m, k n ... -> k m ...', R, merged_weight)
        return rearrange(merged_weight, 'k m ... -> (k m) ...')

    def _process_boft_weight(self, orig_weight: Tensor, R: Tensor, eye: Tensor) -> Tensor:
        """Process weights for BOFT."""
        scale = 1.0  # TODO: determine correct value for scale
        inp = orig_weight
        
        for i in range(self.boft_m):
            bi = R[i] * scale + (1 - scale) * eye if i == 0 else R[i]
            inp = rearrange(inp, "(c g k) ... -> (c k g) ...", g=2, k=2**i * (self.block_size // 2))
            inp = rearrange(inp, "(d b) ... -> d b ...", b=self.block_size)
            inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
            inp = rearrange(inp, "d b ... -> (d b) ...")
            inp = rearrange(inp, "(c k g) ... -> (c g k) ...", g=2, k=2**i * (self.block_size // 2))
            
        return inp