import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class VectorQuantizer2(nn.Module):
    """
    Improved version of VectorQuantizer, designed to be a drop-in replacement.
    This class avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    def __init__(self, n_e: int, e_dim: int, beta: float, remap: str = None,
                 unknown_index: str = "random", sane_index_shape: bool = False,
                 legacy: bool = True):
        """
        Initializes the VectorQuantizer2 instance.

        Args:
            n_e (int): Number of embeddings.
            e_dim (int): Dimension of each embedding.
            beta (float): Weighting factor for the loss.
            remap (str, optional): Path to remapping file. Defaults to None.
            unknown_index (str, optional): Index for unknown entries. Defaults to "random".
            sane_index_shape (bool, optional): If True, returns indices in a specific shape. Defaults to False.
            legacy (bool, optional): If True, uses the legacy behavior for backward compatibility. Defaults to True.
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        self.unknown_index = unknown_index
        self.sane_index_shape = sane_index_shape

        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self._initialize_unknown_index()
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

    def _initialize_unknown_index(self):
        """Initializes the unknown index based on the specified configuration."""
        if self.unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed += 1

    def remap_to_used(self, inds: torch.Tensor) -> torch.Tensor:
        """
        Remaps indices to the used indices.

        Args:
            inds (torch.Tensor): Indices to remap.

        Returns:
            torch.Tensor: Remapped indices.
        """
        ishape = inds.shape
        assert len(ishape) > 1, "Input tensor must have more than one dimension."
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new_indices = match.argmax(-1)
        unknown = match.sum(2) < 1

        if self.unknown_index == "random":
            new_indices[unknown] = torch.randint(0, self.re_embed, size=new_indices[unknown].shape).to(device=new_indices.device)
        else:
            new_indices[unknown] = self.unknown_index

        return new_indices.reshape(ishape)

    def unmap_to_all(self, inds: torch.Tensor) -> torch.Tensor:
        """
        Unmaps indices to all possible indices.

        Args:
            inds (torch.Tensor): Indices to unmap.

        Returns:
            torch.Tensor: Unmapped indices.
        """
        ishape = inds.shape
        assert len(ishape) > 1, "Input tensor must have more than one dimension."
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)

        if self.re_embed > self.used.shape[0]:  # Handle extra token
            inds[inds >= self.used.shape[0]] = 0  # Set to zero for out-of-bounds indices

        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: torch.Tensor, temp: float = None, rescale_logits: bool = False,
                return_logits: bool = False) -> tuple:
        """
        Forward pass for the vector quantizer.

        Args:
            z (torch.Tensor): Input tensor.
            temp (float, optional): Temperature for sampling. Defaults to None.
            rescale_logits (bool, optional): If True, rescales logits. Defaults to False.
            return_logits (bool, optional): If True, returns logits. Defaults to False.

        Returns:
            tuple: Quantized tensor, loss, and additional information.
        """
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert not rescale_logits, "Only for interface compatible with Gumbel"
        assert not return_logits, "Only for interface compatible with Gumbel"

        # Reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # Calculate distances from z to embeddings
        distances = self._calculate_distances(z_flattened)

        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute loss for embedding
        loss = self._compute_loss(z_q, z)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # Add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # Flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (None, None, min_encoding_indices)

    def _calculate_distances(self, z_flattened: torch.Tensor) -> torch.Tensor:
        """Calculates the squared distances from z to embeddings."""
        return (torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                torch.sum(self.embedding.weight ** 2, dim=1) -
                2 * torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n')))

    def _compute_loss(self, z_q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Computes the loss for the embedding."""
        if not self.legacy:
            return (self.beta * torch.mean((z_q.detach() - z) ** 2) +
                    torch.mean((z_q - z.detach()) ** 2))
        else:
            return (torch.mean((z_q.detach() - z) ** 2) +
                    self.beta * torch.mean((z_q - z.detach()) ** 2))

    def get_codebook_entry(self, indices: torch.Tensor, shape: tuple) -> torch.Tensor:
        """
        Retrieves codebook entries based on indices.

        Args:
            indices (torch.Tensor): Indices for the codebook.
            shape (tuple): Shape specifying (batch, height, width, channel).

        Returns:
            torch.Tensor: Codebook entries reshaped to the specified shape.
        """
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # Add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # Flatten again

        # Get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # Reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q