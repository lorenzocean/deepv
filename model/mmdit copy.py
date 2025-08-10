# Modified from:
#   diffusers:    https://github.com/huggingface/diffusers
#   PyramidFlow:  https://github.com/jy0205/Pyramid-Flow


import math
import numbers
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, get_activation
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version
from einops import rearrange
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

# Optional Flash Attention import
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    from flash_attn.bert_padding import index_first_axis, pad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_func = None
    flash_attn_qkvpacked_func = None
    flash_attn_varlen_func = None
    print("Flash Attention is not available. Falling back to standard attention.")


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# |                          Dummy Functions for Compatibility                      |
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# These functions replace the need for `trainer_misc`, which is used for
# distributed training. For single-device inference, we can assume sequence
# parallelism is not initialized.

def is_sequence_parallel_initialized():
    """Returns False as we are not using sequence parallelism for inference."""
    return False

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# |                              HELPER FUNCTIONS                                 |
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Get 1D sinusoidal positional embedding from a grid of positions.

    Args:
        embed_dim (int): Output dimension for each position.
        pos (np.ndarray): A list of positions to be encoded, size (M,).

    Returns:
        np.ndarray: The positional embeddings of shape (M, D).
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, num_frames):
    """Get 1D sinusoidal positional embedding."""
    t = np.arange(num_frames, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, t)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Get 2D sinusoidal positional embedding from a grid."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size, base_size=16, interpolation_scale=1.0):
    """Get 2D sinusoidal positional embedding."""
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """Create sinusoidal timestep embeddings."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """Rotary Position Embedding."""
    assert dim % 2 == 0, "The dimension must be even."
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out, sin_out = torch.cos(out), torch.sin(out)
    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    return stacked_out.view(*pos.shape, -1, dim // 2, 2, 2).float()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# |                      CORE NEURAL NETWORK MODULES & LAYERS                       |
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if is_torch_version(">=", "2.1.0"):
    LayerNorm = nn.LayerNorm
else:
    class LayerNorm(nn.Module):
        """A LayerNorm implementation that supports an optional bias."""
        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            super().__init__()
            self.eps = eps
            self.dim = (dim,) if isinstance(dim, numbers.Integral) else torch.Size(dim)
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight, self.bias = None, None
        def forward(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)

class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.dim = (dim,) if isinstance(dim, numbers.Integral) else torch.Size(dim)
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            hidden_states = (hidden_states.to(self.weight.dtype) * self.weight).to(input_dtype)
        return hidden_states

class FeedForward(nn.Module):
    """A feed-forward layer."""
    def __init__(
        self, dim: int, dim_out: Optional[int] = None, mult: int = 4,
        dropout: float = 0.0, activation_fn: str = "geglu", bias: bool = True
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        self.net = nn.ModuleList([act_fn, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out, bias=bias)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class Timesteps(nn.Module):
    """Module for creating timestep embeddings."""
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
    def forward(self, timesteps):
        return get_timestep_embedding(timesteps, self.num_channels, self.flip_sin_to_cos, self.downscale_freq_shift)

class TimestepEmbedding(nn.Module):
    """Module for embedding timesteps."""
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = get_activation(act_fn)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))

class TextProjection(nn.Module):
    """Module for projecting text embeddings."""
    def __init__(self, in_features, hidden_size, act_fn="silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        self.act_1 = get_activation(act_fn)
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
    def forward(self, caption):
        return self.linear_2(self.act_1(self.linear_1(caption)))

class AdaLayerNormContinuous(nn.Module):
    """Adaptive Layer Normalization with continuous conditioning."""
    def __init__(
        self, embedding_dim: int, conditioning_embedding_dim: int,
        elementwise_affine=True, eps=1e-5, bias=True, norm_type="layer_norm"
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor, hidden_length=None) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        if hidden_length is not None:
            batch_emb = torch.zeros_like(x).repeat(1, 1, 2)
            i_sum, num_stages = 0, len(hidden_length)
            for i_p, length in enumerate(hidden_length):
                batch_emb[:, i_sum:i_sum+length] = emb[i_p::num_stages][:,None]
                i_sum += length
            scale, shift = torch.chunk(batch_emb, 2, dim=2)
        else:
            scale, shift = torch.chunk(emb, 2, dim=1)
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        x = self.norm(x) * (1 + scale) + shift
        return x

class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization Zero (adaLN-Zero)."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, hidden_length=None):
        emb = self.linear(self.silu(emb))
        if hidden_length:
            batch_emb = torch.zeros_like(x).repeat(1, 1, 6)
            i_sum, num_stages = 0, len(hidden_length)
            for i_p, length in enumerate(hidden_length):
                batch_emb[:, i_sum:i_sum+length] = emb[i_p::num_stages][:,None]
                i_sum += length
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = batch_emb.chunk(6, dim=2)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
            shift_msa, scale_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)

        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# |                           ATTENTION MECHANISMS                                |
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class VarlenFlashSelfAttentionWithT5Mask:
    """Flash Attention with variable-length sequences and T5 mask."""
    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def __call__(self, query, key, value, encoder_query, encoder_key, encoder_value,
                 heads, scale, hidden_length=None, image_rotary_emb=None, encoder_attention_mask=None):
        assert encoder_attention_mask is not None
        batch_size = query.shape[0]
        encoder_length = encoder_query.shape[1]
        
        output_hidden = torch.zeros_like(query)
        output_encoder_hidden = torch.zeros_like(encoder_query)

        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2)
        qkv = torch.stack([query, key, value], dim=2)

        qkv_list = []
        i_sum = 0
        num_stages = len(hidden_length)
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])
            
            indices = encoder_attention_mask[i_p]['indices']
            qkv_list.append(index_first_axis(rearrange(concat_qkv_tokens, "b s ... -> (b s) ..."), indices))
            i_sum += length

        token_lengths = [x.shape[0] for x in qkv_list]
        packed_qkv = torch.cat(qkv_list, dim=0)
        query, key, value = packed_qkv.unbind(1)

        cu_seqlens = torch.cat([x['seqlens_in_batch'] for x in encoder_attention_mask], dim=0)
        max_seqlen = cu_seqlens.max().item()
        cu_seqlens_q = F.pad(torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32), (1, 0))

        output = flash_attn_varlen_func(
            query, key, value, cu_seqlens_q, cu_seqlens_q, max_seqlen, max_seqlen,
            dropout_p=0.0, causal=False, softmax_scale=scale
        )

        i_sum = token_sum = 0
        for i_p, length in enumerate(hidden_length):
            stage_output = output[token_sum:token_sum + token_lengths[i_p]]
            stage_output = pad_input(stage_output, encoder_attention_mask[i_p]['indices'], batch_size, encoder_length + length)
            output_hidden[:, i_sum:i_sum+length] = stage_output[:, encoder_length:]
            output_encoder_hidden[i_p::num_stages] = stage_output[:, :encoder_length]
            token_sum += token_lengths[i_p]
            i_sum += length

        return output_hidden.flatten(2, 3), output_encoder_hidden.flatten(2, 3)

class VarlenSelfAttentionWithT5Mask:
    """Standard dot-product Attention with variable-length sequences and T5 mask."""
    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def __call__(self, query, key, value, encoder_query, encoder_key, encoder_value,
                 heads, scale, hidden_length=None, image_rotary_emb=None, attention_mask=None):
        assert attention_mask is not None
        encoder_length = encoder_query.shape[1]
        num_stages = len(hidden_length)
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2)
        qkv = torch.stack([query, key, value], dim=2)
        
        output_encoder_hidden_list, output_hidden_list = [], []
        i_sum = 0
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])
            
            q, k, v = concat_qkv_tokens.unbind(2)
            stage_hidden_states = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, is_causal=False, attn_mask=attention_mask[i_p]
            ).transpose(1, 2).flatten(2, 3)
            
            output_encoder_hidden_list.append(stage_hidden_states[:, :encoder_length])
            output_hidden_list.append(stage_hidden_states[:, encoder_length:])
            i_sum += length

        output_encoder_hidden = rearrange(torch.stack(output_encoder_hidden_list, dim=1), 'b n s d -> (b n) s d')
        output_hidden = torch.cat(output_hidden_list, dim=1)
        return output_hidden, output_encoder_hidden


class JointAttention(nn.Module):
    """Joint Attention module for image and text features."""
    def __init__(
        self, query_dim: int, heads: int = 8, dim_head: int = 64,
        qk_norm: Optional[str] = None, context_pre_only=False,
        use_flash_attn=True, out_dim: int = None, **kwargs
    ):
        super().__init__()
        self.inner_dim = out_dim or dim_head * heads
        self.out_dim = out_dim or query_dim
        self.context_pre_only = context_pre_only
        self.scale = dim_head**-0.5
        self.heads = self.inner_dim // dim_head
        
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=True)

        self.add_k_proj = nn.Linear(query_dim, self.inner_dim)
        self.add_v_proj = nn.Linear(query_dim, self.inner_dim)
        self.add_q_proj = nn.Linear(query_dim, self.inner_dim)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.out_dim, bias=True), nn.Dropout(0.0)])
        if not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=True)

        self.use_flash_attn = use_flash_attn and flash_attn_func is not None
        if self.use_flash_attn:
            self.var_flash_attn = VarlenFlashSelfAttentionWithT5Mask()
        else:
            self.var_len_attn = VarlenSelfAttentionWithT5Mask()

        # QK Normalization
        self.norm_q, self.norm_k, self.norm_add_q, self.norm_add_k = (None,) * 4
        if qk_norm:
            norm_cls = {"layer_norm": nn.LayerNorm, "rms_norm": RMSNorm}[qk_norm]
            self.norm_q = norm_cls(dim_head, eps=1e-5)
            self.norm_k = norm_cls(dim_head, eps=1e-5)
            self.norm_add_q = norm_cls(dim_head, eps=1e-5)
            self.norm_add_k = norm_cls(dim_head, eps=1e-5)

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor, attention_mask: torch.FloatTensor,
        hidden_length: torch.Tensor, image_rotary_emb: torch.Tensor, **kwargs
    ):
        head_dim = self.inner_dim // self.heads
        
        # Project image features
        query = self.to_q(hidden_states).view(hidden_states.shape[0], -1, self.heads, head_dim)
        key = self.to_k(hidden_states).view(hidden_states.shape[0], -1, self.heads, head_dim)
        value = self.to_v(hidden_states).view(hidden_states.shape[0], -1, self.heads, head_dim)

        # Project text features
        encoder_q = self.add_q_proj(encoder_hidden_states).view(encoder_hidden_states.shape[0], -1, self.heads, head_dim)
        encoder_k = self.add_k_proj(encoder_hidden_states).view(encoder_hidden_states.shape[0], -1, self.heads, head_dim)
        encoder_v = self.add_v_proj(encoder_hidden_states).view(encoder_hidden_states.shape[0], -1, self.heads, head_dim)

        # Apply QK norm if configured
        if self.norm_q: query = self.norm_q(query)
        if self.norm_k: key = self.norm_k(key)
        if self.norm_add_q: encoder_q = self.norm_add_q(encoder_q)
        if self.norm_add_k: encoder_k = self.norm_add_k(encoder_k)

        # Joint attention
        if self.use_flash_attn:
            hidden_states, encoder_hidden_states = self.var_flash_attn(
                query, key, value, encoder_q, encoder_k, encoder_v, self.heads,
                self.scale, hidden_length, image_rotary_emb, encoder_attention_mask
            )
        else:
            hidden_states, encoder_hidden_states = self.var_len_attn(
                query, key, value, encoder_q, encoder_k, encoder_v, self.heads,
                self.scale, hidden_length, image_rotary_emb, attention_mask
            )

        # Output projection
        hidden_states = self.to_out[1](self.to_out[0](hidden_states))
        if not self.context_pre_only:
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# |                   TRANSFORMER BLOCK & EMBEDDING LAYERS                          |
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class JointTransformerBlock(nn.Module):
    """A Transformer block from the MMDiT architecture."""
    def __init__(
        self, dim, num_attention_heads, attention_head_dim, qk_norm=None,
        context_pre_only=False, use_flash_attn=True
    ):
        super().__init__()
        self.context_pre_only = context_pre_only
        self.norm1 = AdaLayerNormZero(dim)
        if context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1_context = AdaLayerNormZero(dim)

        self.attn = JointAttention(
            query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim // num_attention_heads,
            out_dim=attention_head_dim, qk_norm=qk_norm, context_pre_only=context_pre_only,
            use_flash_attn=use_flash_attn
        )
        
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor, temb: torch.FloatTensor,
        attention_mask: torch.FloatTensor, hidden_length: List, image_rotary_emb: torch.FloatTensor
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb, hidden_length)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, temb)

        attn_output, context_attn_output = self.attn(
            norm_hidden_states, norm_encoder_hidden_states, encoder_attention_mask,
            attention_mask, hidden_length, image_rotary_emb
        )

        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

        if not self.context_pre_only:
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
            norm_enc = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_enc)
        else:
            encoder_hidden_states = None

        return encoder_hidden_states, hidden_states

class CombinedTimestepConditionEmbeddings(nn.Module):
    """Combines timestep and text condition embeddings."""
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim)
    def forward(self, timestep, pooled_projection):
        timesteps_emb = self.timestep_embedder(self.time_proj(timestep))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections

class EmbedNDRoPE(nn.Module):
    """N-dimensional Rotary Position Embedding."""
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(2)

class PatchEmbed3D(nn.Module):
    """3D Patch Embedding Layer."""
    def __init__(
        self, height=128, width=128, patch_size=2, in_channels=16,
        embed_dim=1536, pos_embed_type="sincos", temp_pos_embed_type='rope',
        pos_embed_max_size=192, max_num_frames=64, add_temp_pos_embed=False, **kwargs
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.proj_history = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.pos_embed_type = pos_embed_type
        self.temp_pos_embed_type = temp_pos_embed_type
        self.pos_embed_max_size = pos_embed_max_size
        self.add_temp_pos_embed = add_temp_pos_embed
        
        if pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(embed_dim, pos_embed_max_size, base_size=height//patch_size)
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=True)
            if add_temp_pos_embed and temp_pos_embed_type == 'sincos':
                time_pos_embed = get_1d_sincos_pos_embed(embed_dim, max_num_frames)
                self.register_buffer("temp_pos_embed", torch.from_numpy(time_pos_embed).float().unsqueeze(0), persistent=True)

    def cropped_pos_embed(self, height, width, ori_height, ori_width):
        """Crops positional embeddings for varying resolutions."""
        h, w = height // self.patch_size, width // self.patch_size
        oh, ow = ori_height // self.patch_size, ori_width // self.patch_size
        top, left = (self.pos_embed_max_size - oh) // 2, (self.pos_embed_max_size - ow) // 2
        
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top:top+oh, left:left+ow, :]
        if oh != h or ow != w:
            spatial_pos_embed = F.interpolate(spatial_pos_embed.permute(0, 3, 1, 2), size=(h, w), mode='bilinear').permute(0, 2, 3, 1)
        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

    def forward_func(self, latent, time_index=0, ori_height=None, ori_width=None):
        bs, _, temp, h, w = latent.shape
        latent = self.proj(rearrange(latent, 'b c t h w -> (b t) c h w')).flatten(2).transpose(1, 2)
        latent = self.norm(latent)
        
        if self.pos_embed_type == 'sincos':
            pos_embed = self.cropped_pos_embed(h, w, ori_height, ori_width)
            latent = latent + pos_embed
            if self.add_temp_pos_embed and self.temp_pos_embed_type == 'sincos':
                latent = rearrange(latent, '(b t) n c -> (b n) t c', t=temp)
                latent = latent + self.temp_pos_embed[:, time_index:time_index + temp, :]
                latent = rearrange(latent, '(b n) t c -> b t n c', b=bs)
            else:
                latent = rearrange(latent, '(b t) n c -> b t n c', b=bs, t=temp)
        else: # rope
            latent = rearrange(latent, '(b t) n c -> b t n c', b=bs, t=temp)
        
        return latent

    def forward(self, latent_pyramid):
        output_list = []
        for latent_levels in latent_pyramid:
            latent_levels = [latent_levels] if not isinstance(latent_levels, list) else latent_levels
            output_latent = []
            time_idx, ori_h, ori_w = 0, latent_levels[-1].shape[-2], latent_levels[-1].shape[-1]
            for each_latent in latent_levels:
                hidden_state = self.forward_func(each_latent, time_index=time_idx, ori_height=ori_h, ori_width=ori_w)
                time_idx += each_latent.shape[2]
                output_latent.append(rearrange(hidden_state, "b t n c -> b (t n) c"))
            output_list.append(torch.cat(output_latent, dim=1))
        return output_list


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# |                          TEXT ENCODER (SD3)                                     |
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class SD3TextEncoderWithMask(nn.Module):
    """The text encoder module for Stable Diffusion 3."""
    def __init__(self, model_path, torch_dtype=torch.float32):
        super().__init__()
        # CLIP-L
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(os.path.join(model_path, 'text_encoder'), torch_dtype=torch_dtype)
        # CLIP-G
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer_2'))
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(os.path.join(model_path, 'text_encoder_2'), torch_dtype=torch_dtype)
        # T5
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(os.path.join(model_path, 'tokenizer_3'))
        self.text_encoder_3 = T5EncoderModel.from_pretrained(os.path.join(model_path, 'text_encoder_3'), torch_dtype=torch_dtype)
    
        self.tokenizer_max_length = self.tokenizer.model_max_length
        self._freeze()

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def _get_t5_prompt_embeds(self, prompt, num_images_per_prompt, device, max_sequence_length=77):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer_3(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.text_encoder_3(text_input_ids, attention_mask=attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_3.dtype, device=device)
        
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(
            batch_size * num_images_per_prompt, prompt_embeds.shape[1], -1
        )
        attention_mask = attention_mask.repeat(num_images_per_prompt, 1)
        return prompt_embeds, attention_mask

    def _get_clip_prompt_embeds(self, prompt, num_images_per_prompt, device, clip_model_index=0):
        tokenizer = [self.tokenizer, self.tokenizer_2][clip_model_index]
        text_encoder = [self.text_encoder, self.text_encoder_2][clip_model_index]
        batch_size = len(prompt)
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer_max_length,
            truncation=True, return_tensors="pt"
        )
        pooled_prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)[0]
        return pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, -1)

    def encode_prompt(self, prompt, num_images_per_prompt=1, device=None):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        pooled_embed_1 = self._get_clip_prompt_embeds(prompt, num_images_per_prompt, device, 0)
        pooled_embed_2 = self._get_clip_prompt_embeds(prompt, num_images_per_prompt, device, 1)
        pooled_prompt_embeds = torch.cat([pooled_embed_1, pooled_embed_2], dim=-1)
        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(prompt, num_images_per_prompt, device)
        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds

    @torch.no_grad()
    def forward(self, prompts, device):
        return self.encode_prompt(prompts, 1, device=device)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# |                             MAIN MODEL: MMDiT                                   |
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class MMDiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self, sample_size: int = 128, patch_size: int = 2, in_channels: int = 16,
        num_layers: int = 24, attention_head_dim: int = 64, num_attention_heads: int = 24,
        caption_projection_dim: int = 1152, pooled_projection_dim: int = 2048,
        pos_embed_max_size: int = 192, max_num_frames: int = 200, qk_norm: str = 'rms_norm',
        pos_embed_type: str = 'rope', temp_pos_embed_type: str = 'sincos',
        joint_attention_dim: int = 4096, use_flash_attn: bool = True,
        use_temporal_causal: bool = False, add_temp_pos_embed: bool = False,
        **kwargs
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.pos_embed_type = pos_embed_type
        self.temp_pos_embed_type = temp_pos_embed_type
        self.add_temp_pos_embed = add_temp_pos_embed
        self.use_flash_attn = use_flash_attn and flash_attn_func is not None
        self.use_temporal_causal = use_temporal_causal
        if use_temporal_causal:
            assert not self.use_flash_attn, "Flash attention does not support temporal causal mask."

        # --- Model Layers ---
        self.pos_embed = PatchEmbed3D(
            height=sample_size, width=sample_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=self.inner_dim, pos_embed_max_size=pos_embed_max_size, max_num_frames=max_num_frames,
            pos_embed_type=pos_embed_type, temp_pos_embed_type=temp_pos_embed_type, add_temp_pos_embed=add_temp_pos_embed
        )
        if pos_embed_type == 'rope':
            self.rope_embed = EmbedNDRoPE(self.inner_dim, 10000, axes_dim=[16, 24, 24])
        if temp_pos_embed_type == 'rope':
            self.temp_rope_embed = EmbedNDRoPE(self.inner_dim, 10000, axes_dim=[attention_head_dim])
        
        self.time_text_embed = CombinedTimestepConditionEmbeddings(self.inner_dim, pooled_projection_dim)
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)
        
        self.transformer_blocks = nn.ModuleList([
            JointTransformerBlock(
                dim=self.inner_dim, num_attention_heads=num_attention_heads,
                attention_head_dim=self.inner_dim, qk_norm=qk_norm,
                context_pre_only=(i == num_layers - 1), use_flash_attn=self.use_flash_attn,
            ) for i in range(num_layers)
        ])
        
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    @torch.no_grad()
    def _prepare_pyramid_latent_image_ids(self, batch_size, temp_list, height_list, width_list, device):
        image_ids_list = []
        base_h, base_w = max(height_list), max(width_list)
        for temp, height, width in zip(temp_list, height_list, width_list):
            h_pos = F.interpolate(torch.arange(base_h)[None,None,:].float(), height, mode='linear').squeeze() if height != base_h else torch.arange(base_h).float()
            w_pos = F.interpolate(torch.arange(base_w)[None,None,:].float(), width, mode='linear').squeeze() if width != base_w else torch.arange(base_w).float()
            
            ids = torch.zeros(temp, height, width, 3, device=device)
            ids[..., 0] += torch.arange(temp, device=device)[:, None, None]
            ids[..., 1] += h_pos[None, :, None]
            ids[..., 2] += w_pos[None, None, :]
            image_ids_list.append(rearrange(ids.expand(batch_size, -1, -1, -1, -1), 'b t h w c -> b (t h w) c'))
        return image_ids_list
    
    @torch.no_grad()
    def _prepare_pyramid_temporal_rope_ids(self, sample, batch_size, device):
        image_ids_list = []
        for sample_ in sample:
            clips = [sample_] if not isinstance(sample_, list) else sample_
            cur_image_ids, start_time_stamp = [], 0
            for clip_ in clips:
                _, _, temp, height, width = clip_.shape
                h, w = height // self.patch_size, width // self.patch_size
                ids = torch.arange(start_time_stamp, start_time_stamp + temp, device=device)[:, None, None]
                ids = ids.expand(-1, h, w).unsqueeze(-1)
                cur_image_ids.append(rearrange(ids.expand(batch_size, -1, -1, -1, -1), 'b t h w c -> b (t h w) c'))
                start_time_stamp += temp
            image_ids_list.append(torch.cat(cur_image_ids, dim=1))
        return image_ids_list

    def merge_input(self, sample, encoder_hidden_length, encoder_attention_mask):
        device         = sample[0][-1].device   if isinstance(sample[0], list) else sample[0].device
        pad_batch_size = sample[0][-1].shape[0] if isinstance(sample[0], list) else sample[0].shape[0]

        temp_list, height_list, width_list, trainable_token_list, hidden_length = [], [], [], [], []
        for s in sample:
            s_ = s[-1] if isinstance(s, list) else s
            _, _, t, h, w = s_.shape
            temp_list.append(t); height_list.append(h//self.patch_size); width_list.append(w//self.patch_size)
            trainable_token_list.append((h//self.patch_size) * (w//self.patch_size) * t)

        image_rotary_emb = None
        if self.temp_pos_embed_type == 'rope' and self.add_temp_pos_embed:
            image_ids_list = self._prepare_pyramid_temporal_rope_ids(sample, pad_batch_size, device)
            text_ids = torch.zeros(pad_batch_size, encoder_attention_mask.shape[1], 1, device=device)
            input_ids_list = [torch.cat([text_ids, image_ids], dim=1) for image_ids in image_ids_list]
            image_rotary_emb = [self.temp_rope_embed(ids) for ids in input_ids_list]

        hidden_states = self.pos_embed(sample)
        hidden_length = [hs.shape[1] for hs in hidden_states]

        if self.use_flash_attn:
            attention_mask, indices_list = None, []
            for i, length in enumerate(hidden_length):
                pad_mask = torch.ones((pad_batch_size, length), dtype=encoder_attention_mask.dtype, device=device)
                full_mask = torch.cat([encoder_attention_mask[i::len(hidden_length)], pad_mask], dim=1)
                indices_list.append({
                    'indices': torch.nonzero(full_mask.flatten(), as_tuple=False).flatten(),
                    'seqlens_in_batch': full_mask.sum(dim=-1, dtype=torch.int32)
                })
            encoder_attention_mask = indices_list
        else:
            attention_mask = []
            for i, length in enumerate(hidden_length):
                # This part is simplified for inference, assuming standard attention masks
                # A proper implementation would construct the mask based on padding
                total_len = encoder_hidden_length + length
                mask = torch.ones(pad_batch_size, 1, total_len, total_len, device=device, dtype=torch.bool)
                attention_mask.append(mask)

        return hidden_states, hidden_length, temp_list, height_list, width_list, trainable_token_list, encoder_attention_mask, attention_mask, image_rotary_emb

    def split_output(self, batch_hidden_states, hidden_length, temps, heights, widths, trainable_token_list):
        output_hidden_list = []
        batch_hidden_states = torch.split(batch_hidden_states, hidden_length, dim=1)
        batch_size = batch_hidden_states[0].shape[0]

        for i, length in enumerate(hidden_length):
            w, h, t = widths[i], heights[i], temps[i]
            hidden = batch_hidden_states[i][:, -trainable_token_list[i]:]
            hidden = hidden.reshape(batch_size, t, h, w, self.patch_size, self.patch_size, self.out_channels)
            hidden = rearrange(hidden, "b t h w p1 p2 c -> b c t (h p1) (w p2)")
            output_hidden_list.append(hidden)
        return output_hidden_list

    def forward(
        self, sample: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor, pooled_projections: torch.FloatTensor,
        timestep_ratio: torch.FloatTensor, **kwargs
    ):
        temb = self.time_text_embed(timestep_ratio, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_length = encoder_hidden_states.shape[1]

        hidden_states, hidden_length, temps, heights, widths, trainable_token_list, \
        encoder_attention_mask, attention_mask, image_rotary_emb = self.merge_input(sample, encoder_hidden_length, encoder_attention_mask)
        
        hidden_states = torch.cat(hidden_states, dim=1)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask, temb=temb,
                attention_mask=attention_mask, hidden_length=hidden_length,
                image_rotary_emb=image_rotary_emb
            )
        
        hidden_states = self.norm_out(hidden_states, temb, hidden_length=hidden_length)
        hidden_states = self.proj_out(hidden_states)
        
        return self.split_output(hidden_states, hidden_length, temps, heights, widths, trainable_token_list)