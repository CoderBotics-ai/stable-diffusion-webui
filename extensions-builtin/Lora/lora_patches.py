from typing import Any
import torch
import torch.nn as nn

import networks
from modules import patches


class LoraPatches:
    """Class to manage LoRA (Low-Rank Adaptation) patches for various PyTorch neural network components."""
    
    def __init__(self) -> None:
        # Linear layer patches
        self.Linear_forward: Any = patches.patch(
            __name__, 
            nn.Linear, 
            'forward', 
            networks.network_Linear_forward
        )
        self.Linear_load_state_dict: Any = patches.patch(
            __name__, 
            nn.Linear, 
            '_load_from_state_dict', 
            networks.network_Linear_load_state_dict
        )
        
        # Convolution layer patches
        self.Conv2d_forward: Any = patches.patch(
            __name__, 
            nn.Conv2d, 
            'forward', 
            networks.network_Conv2d_forward
        )
        self.Conv2d_load_state_dict: Any = patches.patch(
            __name__, 
            nn.Conv2d, 
            '_load_from_state_dict', 
            networks.network_Conv2d_load_state_dict
        )
        
        # Normalization layer patches
        self.GroupNorm_forward: Any = patches.patch(
            __name__, 
            nn.GroupNorm, 
            'forward', 
            networks.network_GroupNorm_forward
        )
        self.GroupNorm_load_state_dict: Any = patches.patch(
            __name__, 
            nn.GroupNorm, 
            '_load_from_state_dict', 
            networks.network_GroupNorm_load_state_dict
        )
        self.LayerNorm_forward: Any = patches.patch(
            __name__, 
            nn.LayerNorm, 
            'forward', 
            networks.network_LayerNorm_forward
        )
        self.LayerNorm_load_state_dict: Any = patches.patch(
            __name__, 
            nn.LayerNorm, 
            '_load_from_state_dict', 
            networks.network_LayerNorm_load_state_dict
        )
        
        # Multi-head attention patches
        self.MultiheadAttention_forward: Any = patches.patch(
            __name__, 
            nn.MultiheadAttention, 
            'forward', 
            networks.network_MultiheadAttention_forward
        )
        self.MultiheadAttention_load_state_dict: Any = patches.patch(
            __name__, 
            nn.MultiheadAttention, 
            '_load_from_state_dict', 
            networks.network_MultiheadAttention_load_state_dict
        )

    def undo(self) -> None:
        """Undo all applied patches and restore original functionality."""
        # Undo Linear layer patches
        self.Linear_forward = patches.undo(__name__, nn.Linear, 'forward')
        self.Linear_load_state_dict = patches.undo(__name__, nn.Linear, '_load_from_state_dict')
        
        # Undo Convolution layer patches
        self.Conv2d_forward = patches.undo(__name__, nn.Conv2d, 'forward')
        self.Conv2d_load_state_dict = patches.undo(__name__, nn.Conv2d, '_load_from_state_dict')
        
        # Undo Normalization layer patches
        self.GroupNorm_forward = patches.undo(__name__, nn.GroupNorm, 'forward')
        self.GroupNorm_load_state_dict = patches.undo(__name__, nn.GroupNorm, '_load_from_state_dict')
        self.LayerNorm_forward = patches.undo(__name__, nn.LayerNorm, 'forward')
        self.LayerNorm_load_state_dict = patches.undo(__name__, nn.LayerNorm, '_load_from_state_dict')
        
        # Undo Multi-head attention patches
        self.MultiheadAttention_forward = patches.undo(__name__, nn.MultiheadAttention, 'forward')
        self.MultiheadAttention_load_state_dict = patches.undo(__name__, nn.MultiheadAttention, '_load_from_state_dict')