from typing import Tuple
import torch
from torch import Tensor


def make_weight_cp(t: Tensor, wa: Tensor, wb: Tensor) -> Tensor:
    """
    Compute weight using CP decomposition.
    
    Args:
        t: Input tensor
        wa: First weight matrix
        wb: Second weight matrix
    
    Returns:
        Tensor: Resulting tensor after CP decomposition
    """
    temp: Tensor = torch.einsum('i j k l, j r -> i r k l', t, wb)
    return torch.einsum('i j k l, i r -> r j k l', temp, wa)


def rebuild_conventional(
    up: Tensor, 
    down: Tensor, 
    shape: Tuple[int, ...], 
    dyn_dim: int | None = None
) -> Tensor:
    """
    Rebuild tensor using conventional method.
    
    Args:
        up: Upper tensor
        down: Lower tensor
        shape: Target shape
        dyn_dim: Dynamic dimension size
    
    Returns:
        Tensor: Rebuilt tensor
    """
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    if dyn_dim is not None:
        up = up[:, :dyn_dim]
        down = down[: dyn_dim, :]
    return (up @ down).reshape(shape)


def rebuild_cp_decomposition(up: Tensor, down: Tensor, mid: Tensor) -> Tensor:
    """
    Rebuild tensor using CP decomposition.
    
    Args:
        up: Upper tensor
        down: Lower tensor
        mid: Middle tensor
    
    Returns:
        Tensor: Rebuilt tensor using CP decomposition
    """
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    return torch.einsum('n m k l, i n, m j -> i j k l', mid, up, down)


def factorization(dimension: int, factor: int = -1) -> Tuple[int, int]:
    '''
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    secon value is a value for weight.

    Because of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    Args:
        dimension: Input dimension to factorize
        factor: Target factor for decomposition (-1 for automatic)

    Returns:
        Tuple[int, int]: Pair of factors (m, n) where m <= n and m * n = dimension

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64
    '''
    if dimension <= 0:
        raise ValueError("Dimension must be positive")

    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n

    if factor < 0:
        factor = dimension

    m, n = 1, dimension
    length = m + n

    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n

    if m > n:
        n, m = m, n

    return m, n