from typing import Optional, Tuple, Union, List
import torch
from torch import Tensor

from modules import devices, rng_philox, shared


def randn(seed: int, shape: Union[Tuple[int, ...], List[int]], generator: Optional[torch.Generator] = None) -> Tensor:
    """Generate a tensor with random numbers from a normal distribution using seed.

    Uses the seed parameter to set the global torch seed; to generate more with that seed, use randn_like/randn_without_seed.
    
    Args:
        seed: Random seed to use
        shape: Shape of the output tensor
        generator: Optional torch Generator instance
    
    Returns:
        Tensor: Random tensor with specified shape
    """
    manual_seed(seed)

    match shared.opts.randn_source:
        case "NV":
            return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)
        case "CPU" if devices.device.type == 'mps':
            return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
        case "CPU":
            return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
        case _:
            return torch.randn(shape, device=devices.device, generator=generator)


def randn_local(seed: int, shape: Union[Tuple[int, ...], List[int]]) -> Tensor:
    """Generate a tensor with random numbers from a normal distribution using seed.

    Does not change the global random number generator. You can only generate the seed's first tensor using this function.
    
    Args:
        seed: Random seed to use
        shape: Shape of the output tensor
    
    Returns:
        Tensor: Random tensor with specified shape
    """
    match shared.opts.randn_source:
        case "NV":
            rng = rng_philox.Generator(seed)
            return torch.asarray(rng.randn(shape), device=devices.device)
        case _:
            local_device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
            local_generator = torch.Generator(local_device).manual_seed(int(seed))
            return torch.randn(shape, device=local_device, generator=local_generator).to(devices.device)


def randn_like(x: Tensor) -> Tensor:
    """Generate a tensor with random numbers from a normal distribution using the previously initialized generator.

    Use either randn() or manual_seed() to initialize the generator.
    
    Args:
        x: Input tensor whose shape and dtype will be matched
    
    Returns:
        Tensor: Random tensor with same shape and dtype as input
    """
    match shared.opts.randn_source:
        case "NV":
            return torch.asarray(nv_rng.randn(x.shape), device=x.device, dtype=x.dtype)
        case "CPU" if x.device.type == 'mps':
            return torch.randn_like(x, device=devices.cpu).to(x.device)
        case _:
            return torch.randn_like(x)


def randn_without_seed(shape: Union[Tuple[int, ...], List[int]], 
                      generator: Optional[torch.Generator] = None) -> Tensor:
    """Generate a tensor with random numbers from a normal distribution using the previously initialized generator.

    Use either randn() or manual_seed() to initialize the generator.
    
    Args:
        shape: Shape of the output tensor
        generator: Optional torch Generator instance
    
    Returns:
        Tensor: Random tensor with specified shape
    """
    match shared.opts.randn_source:
        case "NV":
            return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)
        case "CPU" if devices.device.type == 'mps':
            return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
        case _:
            return torch.randn(shape, device=devices.device, generator=generator)


def manual_seed(seed: int) -> None:
    """Set up a global random number generator using the specified seed.
    
    Args:
        seed: Random seed to use
    """
    if shared.opts.randn_source == "NV":
        global nv_rng
        nv_rng = rng_philox.Generator(seed)
        return

    torch.manual_seed(seed)


def create_generator(seed: int) -> Union[rng_philox.Generator, torch.Generator]:
    """Create a random number generator with the specified seed.
    
    Args:
        seed: Random seed to use
    
    Returns:
        Union[rng_philox.Generator, torch.Generator]: New random number generator
    """
    if shared.opts.randn_source == "NV":
        return rng_philox.Generator(seed)

    device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    return torch.Generator(device).manual_seed(int(seed))


def slerp(val: float, low: Tensor, high: Tensor) -> Tensor:
    """Spherical linear interpolation between two tensors.
    
    Args:
        val: Interpolation factor between 0 and 1
        low: First tensor
        high: Second tensor
    
    Returns:
        Tensor: Interpolated tensor
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


class ImageRNG:
    def __init__(self, 
                 shape: Tuple[int, ...],
                 seeds: List[int],
                 subseeds: Optional[List[int]] = None,
                 subseed_strength: float = 0.0,
                 seed_resize_from_h: int = 0,
                 seed_resize_from_w: int = 0):
        """Initialize ImageRNG with given parameters.
        
        Args:
            shape: Shape of the output tensors
            seeds: List of random seeds
            subseeds: Optional list of secondary seeds
            subseed_strength: Strength of subseed influence
            seed_resize_from_h: Original height for resize
            seed_resize_from_w: Original width for resize
        """
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w

        self.generators = [create_generator(seed) for seed in seeds]
        self.is_first = True

    def first(self) -> Tensor:
        """Generate the first batch of random tensors."""
        noise_shape = (
            self.shape if self.seed_resize_from_h <= 0 or self.seed_resize_from_w <= 0 
            else (self.shape[0], int(self.seed_resize_from_h) // 8, int(self.seed_resize_from_w // 8))
        )

        xs = []

        for i, (seed, generator) in enumerate(zip(self.seeds, self.generators)):
            subnoise = None
            if self.subseeds is not None and self.subseed_strength != 0:
                subseed = 0 if i >= len(self.subseeds) else self.subseeds[i]
                subnoise = randn(subseed, noise_shape)

            noise = (
                randn(seed, noise_shape) if noise_shape != self.shape
                else randn(seed, self.shape, generator=generator)
            )

            if subnoise is not None:
                noise = slerp(self.subseed_strength, noise, subnoise)

            if noise_shape != self.shape:
                x = randn(seed, self.shape, generator=generator)
                dx = (self.shape[2] - noise_shape[2]) // 2
                dy = (self.shape[1] - noise_shape[1]) // 2
                w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
                h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
                tx = 0 if dx < 0 else dx
                ty = 0 if dy < 0 else dy
                dx = max(-dx, 0)
                dy = max(-dy, 0)

                x[:, ty:ty + h, tx:tx + w] = noise[:, dy:dy + h, dx:dx + w]
                noise = x

            xs.append(noise)

        eta_noise_seed_delta = shared.opts.eta_noise_seed_delta or 0
        if eta_noise_seed_delta:
            self.generators = [create_generator(seed + eta_noise_seed_delta) for seed in self.seeds]

        return torch.stack(xs).to(shared.device)

    def next(self) -> Tensor:
        """Generate subsequent batches of random tensors."""
        if self.is_first:
            self.is_first = False
            return self.first()

        xs = [randn_without_seed(self.shape, generator=generator) for generator in self.generators]
        return torch.stack(xs).to(shared.device)


# Register functions with devices module
devices.randn = randn
devices.randn_local = randn_local
devices.randn_like = randn_like
devices.randn_without_seed = randn_without_seed
devices.manual_seed = manual_seed