from __future__ import annotations
import math
import psutil
import platform

import torch
from torch import einsum

from ldm.util import default
from einops import rearrange

from modules import shared, errors, devices, sub_quadratic_attention
from modules.hypernetworks import hypernetwork

import ldm.modules.attention
import ldm.modules.diffusionmodules.model

import sgm.modules.attention
import sgm.modules.diffusionmodules.model

diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward
sgm_diffusionmodules_model_AttnBlock_forward = sgm.modules.diffusionmodules.model.AttnBlock.forward


class SdOptimization:
    name: str = None
    label: str | None = None
    cmd_opt: str | None = None
    priority: int = 0

    def title(self) -> str:
        return f"{self.name} - {self.label}" if self.label else self.name

    def is_available(self) -> bool:
        return True

    def apply(self) -> None:
        pass

    def undo(self) -> None:
        ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward

        sgm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sgm_diffusionmodules_model_AttnBlock_forward


class SdOptimizationXformers(SdOptimization):
    name = "xformers"
    cmd_opt = "xformers"
    priority = 100

    def is_available(self) -> bool:
        return shared.cmd_opts.force_enable_xformers or (
            shared.xformers_available and 
            torch.cuda.is_available() and 
            (6, 0) <= torch.cuda.get_device_capability(shared.device) <= (9, 0)
        )

    def apply(self) -> None:
        ldm.modules.attention.CrossAttention.forward = xformers_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = xformers_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = xformers_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = xformers_attnblock_forward


class SdOptimizationSdpNoMem(SdOptimization):
    name = "sdp-no-mem"
    label = "scaled dot product without memory efficient attention"
    cmd_opt = "opt_sdp_no_mem_attention"
    priority = 80

    def is_available(self) -> bool:
        return hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(torch.nn.functional.scaled_dot_product_attention)

    def apply(self) -> None:
        ldm.modules.attention.CrossAttention.forward = scaled_dot_product_no_mem_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sdp_no_mem_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = scaled_dot_product_no_mem_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sdp_no_mem_attnblock_forward


class SdOptimizationSdp(SdOptimizationSdpNoMem):
    name = "sdp"
    label = "scaled dot product"
    cmd_opt = "opt_sdp_attention"
    priority = 70

    def apply(self) -> None:
        ldm.modules.attention.CrossAttention.forward = scaled_dot_product_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sdp_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = scaled_dot_product_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sdp_attnblock_forward


class SdOptimizationSubQuad(SdOptimization):
    name = "sub-quadratic"
    cmd_opt = "opt_sub_quad_attention"

    @property
    def priority(self) -> int:
        return 1000 if shared.device.type == 'mps' else 10

    def apply(self) -> None:
        ldm.modules.attention.CrossAttention.forward = sub_quad_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sub_quad_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = sub_quad_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sub_quad_attnblock_forward


class SdOptimizationV1(SdOptimization):
    name = "V1"
    label = "original v1"
    cmd_opt = "opt_split_attention_v1"
    priority = 10

    def apply(self) -> None:
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_v1
        sgm.modules.attention.CrossAttention.forward = split_cross_attention_forward_v1


class SdOptimizationInvokeAI(SdOptimization):
    name = "InvokeAI"
    cmd_opt = "opt_split_attention_invokeai"

    @property
    def priority(self) -> int:
        return 1000 if shared.device.type != 'mps' and not torch.cuda.is_available() else 10

    def apply(self) -> None:
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_invokeAI
        sgm.modules.attention.CrossAttention.forward = split_cross_attention_forward_invokeAI


class SdOptimizationDoggettx(SdOptimization):
    name = "Doggettx"
    cmd_opt = "opt_split_attention"
    priority = 90

    def apply(self) -> None:
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = split_cross_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward


def list_optimizers(res: list) -> None:
    res.extend([
        SdOptimizationXformers(),
        SdOptimizationSdpNoMem(),
        SdOptimizationSdp(),
        SdOptimizationSubQuad(),
        SdOptimizationV1(),
        SdOptimizationInvokeAI(),
        SdOptimizationDoggettx(),
    ])


if shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers:
    try:
        import xformers.ops
        shared.xformers_available = True
    except Exception:
        errors.report("Cannot import xformers", exc_info=True)


def get_available_vram() -> int:
    if shared.device.type == 'cuda':
        stats = torch.cuda.memory_stats(shared.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        return mem_free_total
    else:
        return psutil.virtual_memory().available


def split_cross_attention_forward_v1(self, x, context=None, mask=None, **kwargs):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)
    del context, context_k, context_v, x

    q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q_in, k_in, v_in))
    del q_in, k_in, v_in

    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[0], 2):
            end = i + 2
            s1 = einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
            s1 *= self.scale

            s2 = s1.softmax(dim=-1)
            del s1

            r1[i:end] = einsum('b i j, b j d -> b i d', s2, v[i:end])
            del s2
        del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


def split_cross_attention_forward(self, x, context=None, mask=None, **kwargs):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    dtype = q_in.dtype
    if shared.opts.upcast_attn:
        q_in, k_in, v_in = q_in.float(), k_in.float(), v_in if v_in.device.type == 'mps' else v_in.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k_in = k_in * self.scale

        del context, x

        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q_in, k_in, v_in))
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        mem_free_total = get_available_vram()

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

        slice_size = q.shape[1] // steps
        for i in range(0, q.shape[1], slice_size):
            end = min(i + slice_size, q.shape[1])
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2

        del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


mem_total_gb = psutil.virtual_memory().total // (1 << 30)


def einsum_op_compvis(q, k, v):
    s = einsum('b i d, b j d -> b i j', q, k)
    s = s.softmax(dim=-1, dtype=s.dtype)
    return einsum('b i j, b j d -> b i d', s, v)


def einsum_op_slice_0(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[0], slice_size):
        end = i + slice_size
        r[i:end] = einsum_op_compvis(q[i:end], k[i:end], v[i:end])
    return r


def einsum_op_slice_1(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        r[:, i:end] = einsum_op_compvis(q[:, i:end], k, v)
    return r


def einsum_op_mps_v1(q, k, v):
    if q.shape[0] * q.shape[1] <= 2**16: # (512x512) max q.shape[1]: 4096
        return einsum_op_compvis(q, k, v)
    else:
        slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
        if slice_size % 4096 == 0:
            slice_size -= 1
        return einsum_op_slice_1(q, k, v, slice_size)


def einsum_op_mps_v2(q, k, v):
    if mem_total_gb > 8 and q.shape[0] * q.shape[1] <= 2**16:
        return einsum_op_compvis(q, k, v)
    else:
        return einsum_op_slice_0(q, k, v, 1)


def einsum_op_tensor_mem(q, k, v, max_tensor_mb):
    size_mb = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size() // (1 << 20)
    if size_mb <= max_tensor_mb:
        return einsum_op_compvis(q, k, v)
    div = 1 << int((size_mb - 1) / max_tensor_mb).bit_length()
    if div <= q.shape[0]:
        return einsum_op_slice_0(q, k, v, q.shape[0] // div)
    return einsum_op_slice_1(q, k, v, max(q.shape[1] // div, 1))


def einsum_op_cuda(q, k, v):
    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(q.device)
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch
    return einsum_op_tensor_mem(q, k, v, mem_free_total / 3.3 / (1 << 20))


def einsum_op(q, k, v):
    if q.device.type == 'cuda':
        return einsum_op_cuda(q, k, v)

    if q.device.type == 'mps':
        if mem_total_gb >= 32 and q.shape[0] % 32 != 0 and q.shape[0] * q.shape[1] < 2**18:
            return einsum_op_mps_v1(q, k, v)
        return einsum_op_mps_v2(q, k, v)

    return einsum_op_tensor_mem(q, k, v, 32)


def split_cross_attention_forward_invokeAI(self, x, context=None, mask=None, **kwargs):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, context_k, context_v, x

    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k = k * self.scale

        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q, k, v))
        r = einsum_op(q, k, v)
    r = r.to(dtype)
    return self.to_out(rearrange(r, '(b h) n d -> b n (h d)', h=h))