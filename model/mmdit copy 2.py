import torch
import torch.nn as nn
import os
import torch.nn.functional as F

from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version
from typing import Any, Callable, Dict, List, Optional, Union
from typing import Dict, Optional, Tuple, List


from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from diffusers.models.activations import get_activation
from einops import rearrange

from diffusers.utils import is_torch_version
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except:
    flash_attn_func = None
    flash_attn_qkvpacked_func = None
    flash_attn_varlen_func = None

# from .modeling_embedding import PatchEmbed3D, CombinedTimestepConditionEmbeddings
# from .modeling_normalization import AdaLayerNormContinuous
# from .modeling_mmdit_block import JointTransformerBlock

# from trainer_misc import (
#     is_sequence_parallel_initialized,
#     get_sequence_parallel_group,
#     get_sequence_parallel_world_size,
#     get_sequence_parallel_rank,
#     all_to_all,
# )

def is_sequence_parallel_initialized():
    return False

from IPython import embed
import numbers


if is_torch_version(">=", "2.1.0"):
    LayerNorm = nn.LayerNorm
else:
    # Has optional bias parameter compared to torch layer norm
    # TODO: replace with torch layernorm once min required torch version >= 2.1
    class LayerNorm(nn.Module):
        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            super().__init__()

            self.eps = eps

            if isinstance(dim, numbers.Integral):
                dim = (dim,)

            self.dim = torch.Size(dim)

            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight = None
                self.bias = None

        def forward(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)



class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class VarlenFlashSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, encoder_attention_mask=None,
        ):
        assert encoder_attention_mask is not None, "The encoder-hidden mask needed to be set"

        batch_size = query.shape[0]
        output_hidden = torch.zeros_like(query)
        output_encoder_hidden = torch.zeros_like(encoder_query)
        encoder_length = encoder_query.shape[1]

        qkv_list = []
        num_stages = len(hidden_length)        
    
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, tot_seq, 3, nhead, dim]
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            indices = encoder_attention_mask[i_p]['indices']
            qkv_list.append(index_first_axis(rearrange(concat_qkv_tokens, "b s ... -> (b s) ..."), indices))
            i_sum += length

        token_lengths = [x_.shape[0] for x_ in qkv_list]
        qkv = torch.cat(qkv_list, dim=0)
        query, key, value = qkv.unbind(1)

        cu_seqlens = torch.cat([x_['seqlens_in_batch'] for x_ in encoder_attention_mask], dim=0)
        max_seqlen_q = cu_seqlens.max().item()
        max_seqlen_k = max_seqlen_q
        cu_seqlens_q = F.pad(torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32), (1, 0))
        cu_seqlens_k = cu_seqlens_q.clone()

        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )

        # To merge the tokens
        i_sum = 0;token_sum = 0
        for i_p, length in enumerate(hidden_length):
            tot_token_num = token_lengths[i_p]
            stage_output = output[token_sum : token_sum + tot_token_num]
            stage_output = pad_input(stage_output, encoder_attention_mask[i_p]['indices'], batch_size, encoder_length + length)
            stage_encoder_hidden_output = stage_output[:, :encoder_length]
            stage_hidden_output = stage_output[:, encoder_length:]   
            output_hidden[:, i_sum:i_sum+length] = stage_hidden_output
            output_encoder_hidden[i_p::num_stages] = stage_encoder_hidden_output
            token_sum += tot_token_num
            i_sum += length

        output_hidden = output_hidden.flatten(2, 3)
        output_encoder_hidden = output_encoder_hidden.flatten(2, 3)

        return output_hidden, output_encoder_hidden


class SequenceParallelVarlenFlashSelfAttentionWithT5Mask:

    def __init__(self):
        pass

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, encoder_attention_mask=None,
        ):
        assert encoder_attention_mask is not None, "The encoder-hidden mask needed to be set"

        batch_size = query.shape[0]
        qkv_list = []
        num_stages = len(hidden_length)

        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        # To sync the encoder query, key and values
        sp_group = get_sequence_parallel_group()
        sp_group_size = get_sequence_parallel_world_size()
        encoder_qkv = all_to_all(encoder_qkv, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]

        output_hidden = torch.zeros_like(qkv[:,:,0])
        output_encoder_hidden = torch.zeros_like(encoder_qkv[:,:,0])
        encoder_length = encoder_qkv.shape[1]
        
        i_sum = 0
        for i_p, length in enumerate(hidden_length):
            # get the query, key, value from padding sequence
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            qkv_tokens = all_to_all(qkv_tokens, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, pad_seq, 3, nhead, dim]

            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            indices = encoder_attention_mask[i_p]['indices']
            qkv_list.append(index_first_axis(rearrange(concat_qkv_tokens, "b s ... -> (b s) ..."), indices))
            i_sum += length

        token_lengths = [x_.shape[0] for x_ in qkv_list]
        qkv = torch.cat(qkv_list, dim=0)
        query, key, value = qkv.unbind(1)

        cu_seqlens = torch.cat([x_['seqlens_in_batch'] for x_ in encoder_attention_mask], dim=0)
        max_seqlen_q = cu_seqlens.max().item()
        max_seqlen_k = max_seqlen_q
        cu_seqlens_q = F.pad(torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32), (1, 0))
        cu_seqlens_k = cu_seqlens_q.clone()

        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )

        # To merge the tokens
        i_sum = 0;token_sum = 0
        for i_p, length in enumerate(hidden_length):
            tot_token_num = token_lengths[i_p]
            stage_output = output[token_sum : token_sum + tot_token_num]
            stage_output = pad_input(stage_output, encoder_attention_mask[i_p]['indices'], batch_size, encoder_length + length * sp_group_size)
            stage_encoder_hidden_output = stage_output[:, :encoder_length]
            stage_hidden_output = stage_output[:, encoder_length:]
            stage_hidden_output = all_to_all(stage_hidden_output, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
            output_hidden[:, i_sum:i_sum+length] = stage_hidden_output
            output_encoder_hidden[i_p::num_stages] = stage_encoder_hidden_output
            token_sum += tot_token_num
            i_sum += length

        output_encoder_hidden = all_to_all(output_encoder_hidden, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
        output_hidden = output_hidden.flatten(2, 3)
        output_encoder_hidden = output_encoder_hidden.flatten(2, 3)

        return output_hidden, output_encoder_hidden


class VarlenSelfAttentionWithT5Mask:

    """
        For chunk stage attention without using flash attention
    """

    def __init__(self):
        pass

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, attention_mask=None,
        ):
        assert attention_mask is not None, "The attention mask needed to be set"

        encoder_length = encoder_query.shape[1]
        num_stages = len(hidden_length)        
    
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        i_sum = 0
        output_encoder_hidden_list = []
        output_hidden_list = []
    
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, tot_seq, 3, nhead, dim]
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            query, key, value = concat_qkv_tokens.unbind(2)   # [bs, tot_seq, nhead, dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
            stage_hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask[i_p],
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2).flatten(2, 3)   # [bs, tot_seq, dim]

            output_encoder_hidden_list.append(stage_hidden_states[:, :encoder_length])
            output_hidden_list.append(stage_hidden_states[:, encoder_length:])
            i_sum += length

        output_encoder_hidden = torch.stack(output_encoder_hidden_list, dim=1)  # [b n s d]
        output_encoder_hidden = rearrange(output_encoder_hidden, 'b n s d -> (b n) s d')
        output_hidden = torch.cat(output_hidden_list, dim=1)

        return output_hidden, output_encoder_hidden


class SequenceParallelVarlenSelfAttentionWithT5Mask:
    """
        For chunk stage attention without using flash attention
    """

    def __init__(self):
        pass

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def __call__(
            self, query, key, value, encoder_query, encoder_key, encoder_value, 
            heads, scale, hidden_length=None, image_rotary_emb=None, attention_mask=None,
        ):
        assert attention_mask is not None, "The attention mask needed to be set"

        num_stages = len(hidden_length)        
    
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2) # [bs, sub_seq, 3, head, head_dim]
        qkv = torch.stack([query, key, value], dim=2) # [bs, sub_seq, 3, head, head_dim]

        # To sync the encoder query, key and values
        sp_group = get_sequence_parallel_group()
        sp_group_size = get_sequence_parallel_world_size()
        encoder_qkv = all_to_all(encoder_qkv, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]
        encoder_length = encoder_qkv.shape[1]

        i_sum = 0
        output_encoder_hidden_list = []
        output_hidden_list = []
    
        for i_p, length in enumerate(hidden_length):
            encoder_qkv_tokens = encoder_qkv[i_p::num_stages]
            qkv_tokens = qkv[:, i_sum:i_sum+length]
            qkv_tokens = all_to_all(qkv_tokens, sp_group, sp_group_size, scatter_dim=3, gather_dim=1) # [bs, seq, 3, sub_head, head_dim]
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)  # [bs, tot_seq, 3, nhead, dim]
            
            if image_rotary_emb is not None:
                concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1] = self.apply_rope(concat_qkv_tokens[:,:,0], concat_qkv_tokens[:,:,1], image_rotary_emb[i_p])

            query, key, value = concat_qkv_tokens.unbind(2)   # [bs, tot_seq, nhead, dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            stage_hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask[i_p],
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2)   # [bs, tot_seq, nhead, dim]

            output_encoder_hidden_list.append(stage_hidden_states[:, :encoder_length])

            output_hidden = stage_hidden_states[:, encoder_length:]
            output_hidden = all_to_all(output_hidden, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
            output_hidden_list.append(output_hidden)

            i_sum += length

        output_encoder_hidden = torch.stack(output_encoder_hidden_list, dim=1)  # [b n s nhead d]
        output_encoder_hidden = rearrange(output_encoder_hidden, 'b n s h d -> (b n) s h d')
        output_encoder_hidden = all_to_all(output_encoder_hidden, sp_group, sp_group_size, scatter_dim=1, gather_dim=2)
        output_encoder_hidden = output_encoder_hidden.flatten(2, 3)
        output_hidden = torch.cat(output_hidden_list, dim=1).flatten(2, 3)

        return output_hidden, output_encoder_hidden


class JointAttention(nn.Module):
    
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only=None,
        use_flash_attn=True,
    ): 
        """
            Fixing the QKNorm, following the flux, norm the head dimension
        """
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.use_bias = bias
        self.dropout = dropout

        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only

        self.scale = dim_head**-0.5
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)
        elif qk_norm == 'rms_norm':
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
    
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)

            if qk_norm is None:
                self.norm_add_q = None
                self.norm_add_k = None
            elif qk_norm == "layer_norm":
                self.norm_add_q = nn.LayerNorm(dim_head, eps=eps)
                self.norm_add_k = nn.LayerNorm(dim_head, eps=eps)
            elif qk_norm == 'rms_norm':
                self.norm_add_q = RMSNorm(dim_head, eps=eps)
                self.norm_add_k = RMSNorm(dim_head, eps=eps)
            else:
                raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        self.use_flash_attn = use_flash_attn

        if flash_attn_func is None:
            self.use_flash_attn = False

        # print(f"Using flash-attention: {self.use_flash_attn}")
        if self.use_flash_attn:
            if is_sequence_parallel_initialized():
                self.var_flash_attn = SequenceParallelVarlenFlashSelfAttentionWithT5Mask()
            else:
                self.var_flash_attn = VarlenFlashSelfAttentionWithT5Mask()
        else:
            if is_sequence_parallel_initialized():
                self.var_len_attn = SequenceParallelVarlenSelfAttentionWithT5Mask()
            else:
                self.var_len_attn = VarlenSelfAttentionWithT5Mask()
    

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,   # [B, L, S]
        hidden_length: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        # This function is only used during training
        # `sample` projections.
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(query.shape[0], -1, self.heads, head_dim)
        key = key.view(key.shape[0], -1, self.heads, head_dim)
        value = value.view(value.shape[0], -1, self.heads, head_dim)

        if self.norm_q is not None:
            query = self.norm_q(query)

        if self.norm_k is not None:
            key = self.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj   = self.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            encoder_hidden_states_query_proj.shape[0], -1, self.heads, head_dim
        )
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            encoder_hidden_states_key_proj.shape[0], -1, self.heads, head_dim
        )
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            encoder_hidden_states_value_proj.shape[0], -1, self.heads, head_dim
        )

        if self.norm_add_q is not None:
            encoder_hidden_states_query_proj = self.norm_add_q(encoder_hidden_states_query_proj)

        if self.norm_add_k is not None:
            encoder_hidden_states_key_proj = self.norm_add_k(encoder_hidden_states_key_proj)

        # To cat the hidden and encoder hidden, perform attention compuataion, and then split
        if self.use_flash_attn:
            hidden_states, encoder_hidden_states = self.var_flash_attn(
                query, key, value, 
                encoder_hidden_states_query_proj, encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj, self.heads, self.scale, hidden_length, 
                image_rotary_emb, encoder_attention_mask,
            )
        else:
            hidden_states, encoder_hidden_states = self.var_len_attn(
                query, key, value, 
                encoder_hidden_states_query_proj, encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj, self.heads, self.scale, hidden_length, 
                image_rotary_emb, attention_mask,
            )

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        if not self.context_pre_only:
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class JointTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self, dim, num_attention_heads, attention_head_dim, qk_norm=None, 
        context_pre_only=False, use_flash_attn=True,
    ):
        super().__init__()

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        self.attn = JointAttention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim // num_attention_heads,
            heads=num_attention_heads,
            out_dim=attention_head_dim,
            qk_norm=qk_norm,
            context_pre_only=context_pre_only,
            bias=True,
            use_flash_attn=use_flash_attn,
        )
        
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, 
        encoder_attention_mask: torch.FloatTensor, temb: torch.FloatTensor, 
        attention_mask: torch.FloatTensor = None, hidden_length: List = None, 
        image_rotary_emb: torch.FloatTensor = None,
    ):
        # print(f'[transformer info] {hidden_states.shape, encoder_hidden_states.shape, encoder_attention_mask.shape, temb.shape, hidden_length}')

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb, hidden_length=hidden_length)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb,
            )

        # Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, attention_mask=attention_mask, 
            hidden_length=hidden_length, image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight

        hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
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

    def forward_with_pad(self, x: torch.Tensor, conditioning_embedding: torch.Tensor, hidden_length=None) -> torch.Tensor:
        assert hidden_length is not None
        
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        batch_emb = torch.zeros_like(x).repeat(1, 1, 2)

        i_sum = 0
        num_stages = len(hidden_length)
        for i_p, length in enumerate(hidden_length):
            batch_emb[:, i_sum:i_sum+length] = emb[i_p::num_stages][:,None]
            i_sum += length

        batch_scale, batch_shift = torch.chunk(batch_emb, 2, dim=2)
        x = self.norm(x) * (1 + batch_scale) + batch_shift
        return x

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor, hidden_length=None) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        if hidden_length is not None:
            return self.forward_with_pad(x, conditioning_embedding, hidden_length)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None):
        super().__init__()
        self.emb = None
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)


    def forward_with_pad(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
        hidden_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [bs, seq_len, dim]
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)

        emb = self.linear(self.silu(emb))
        batch_emb = torch.zeros_like(x).repeat(1, 1, 6)
    
        i_sum = 0
        num_stages = len(hidden_length)
        for i_p, length in enumerate(hidden_length):
            batch_emb[:, i_sum:i_sum+length] = emb[i_p::num_stages][:,None]
            i_sum += length

        batch_shift_msa, batch_scale_msa, batch_gate_msa, batch_shift_mlp, batch_scale_mlp, batch_gate_mlp = batch_emb.chunk(6, dim=2)
        x = self.norm(x) * (1 + batch_scale_msa) + batch_shift_msa
        return x, batch_gate_msa, batch_shift_mlp, batch_scale_mlp, batch_gate_mlp


    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
        hidden_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_length is not None:
            return self.forward_with_pad(x, timestep, class_labels, hidden_dtype, emb, hidden_length)
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

def get_1d_sincos_pos_embed(
    embed_dim, num_frames, cls_token=False, extra_tokens=0,
):
    t = np.arange(num_frames, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, t)  # (T, D)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        sample_proj_bias=True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.act = get_activation(act_fn)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, sample_proj_bias)

    def forward(self, sample):
        # if self.linear_1.weight.ndim == 1:
        #     self.linear_1.weight = self.linear_1.weight.view(-1, sample.shape[-1])
        # print(f'[TimestepEmbedding info] {sample.shape, self.linear_1.weight.shape, self.linear_1.weight.dtype, self.linear_1.weight.device, self.linear_1.bias.shape}')

        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, act_fn="silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        self.act_1 = get_activation(act_fn)
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepConditionEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        # timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = timesteps_emb + pooled_projections
        return conditioning


class CombinedTimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class PatchEmbed3D(nn.Module):
    """Support the 3D Tensor input"""

    def __init__(
        self,
        height=128,
        width=128,
        patch_size=2,
        in_channels=16,
        embed_dim=1536,
        layer_norm=False,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        temp_pos_embed_type='rope',
        pos_embed_max_size=192,   # For SD3 cropping
        max_num_frames=64,
        add_temp_pos_embed=False,
        interp_condition_pos=False,
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        self.add_temp_pos_embed = add_temp_pos_embed

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None

        elif pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim, grid_size, base_size=self.base_size, interpolation_scale=self.interpolation_scale
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=persistent)

            if add_temp_pos_embed and temp_pos_embed_type == 'sincos':
                time_pos_embed = get_1d_sincos_pos_embed(embed_dim, max_num_frames)
                self.register_buffer("temp_pos_embed", torch.from_numpy(time_pos_embed).float().unsqueeze(0), persistent=True)

        elif pos_embed_type == "rope":
            print("Using the rotary position embedding")

        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

        self.pos_embed_type = pos_embed_type
        self.temp_pos_embed_type = temp_pos_embed_type
        self.interp_condition_pos = interp_condition_pos

        # record
        # self.history_in_channels = in_channels
        # self.history_embed_dim   = embed_dim
        self.history_patch_size  = patch_size
        self.history_bias        = bias
    
    def add_history_modules(self):
        self.history_in_channels = self.proj.in_channels
        self.history_embed_dim   = self.proj.out_channels
        # self.history_patch_size  = self.proj.stride
        # self.history_bias        = self.proj.bias

        print(f'[debug patch] {self.history_in_channels, self.history_embed_dim, self.history_patch_size, self.history_bias}')

        self.proj_history = nn.Conv2d(
            self.history_in_channels, self.history_embed_dim, kernel_size=(self.history_patch_size, self.history_patch_size), stride=self.history_patch_size, bias=self.history_bias
        )
        if self.layer_norm:
            self.norm_history = nn.LayerNorm(self.history_embed_dim, elementwise_affine=False, eps=1e-6)
        



    def cropped_pos_embed(self, height, width, ori_height, ori_width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        ori_height = ori_height // self.patch_size
        ori_width = ori_width // self.patch_size

        assert ori_height >= height, "The ori_height needs >= height"
        assert ori_width >= width, "The ori_width needs >= width"

        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        if self.interp_condition_pos:
            top = (self.pos_embed_max_size - ori_height) // 2
            left = (self.pos_embed_max_size - ori_width) // 2
            spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
            spatial_pos_embed = spatial_pos_embed[:, top : top + ori_height, left : left + ori_width, :]   # [b h w c]
            if ori_height != height or ori_width != width:
                spatial_pos_embed = spatial_pos_embed.permute(0, 3, 1, 2)
                spatial_pos_embed = torch.nn.functional.interpolate(spatial_pos_embed, size=(height, width), mode='bilinear')
                spatial_pos_embed = spatial_pos_embed.permute(0, 2, 3, 1)
        else:
            top = (self.pos_embed_max_size - height) // 2
            left = (self.pos_embed_max_size - width) // 2
            spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
            spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

        return spatial_pos_embed

    def forward_func(self, latent, time_index=0, ori_height=None, ori_width=None, history=None):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        bs   = latent.shape[0]
        temp = latent.shape[2]

        # before_shape = latent.shape
        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        if history is None:
            latent = self.proj(latent)
        else:
            latent = self.proj_history(latent)

        # after_shape = latent.shape
        latent = latent.flatten(2).transpose(1, 2)  # (BT)CHW -> (BT)NC
        # print(f'[shape trans] {before_shape} -> {after_shape}')

        if self.layer_norm:
            if history is None:
                latent = self.norm(latent)
            else:
                latent = self.norm_history(latent)

        if self.pos_embed_type == 'sincos':
            # Spatial position embedding, Interpolate or crop positional embeddings as needed
            if self.pos_embed_max_size:
                pos_embed = self.cropped_pos_embed(height, width, ori_height, ori_width)
            else:
                raise NotImplementedError("Not implemented sincos pos embed without sd3 max pos crop")
                if self.height != height or self.width != width:
                    pos_embed = get_2d_sincos_pos_embed(
                        embed_dim=self.pos_embed.shape[-1],
                        grid_size=(height, width),
                        base_size=self.base_size,
                        interpolation_scale=self.interpolation_scale,
                    )
                    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
                else:
                    pos_embed = self.pos_embed

            if self.add_temp_pos_embed and self.temp_pos_embed_type == 'sincos':
                latent_dtype = latent.dtype
                if history is None:
                    latent = latent + pos_embed
                latent = rearrange(latent, '(b t) n c -> (b n) t c', t=temp)
                if history is None:
                    latent = latent + self.temp_pos_embed[:, time_index:time_index + temp, :]
                latent = latent.to(latent_dtype)
                latent = rearrange(latent, '(b n) t c -> b t n c', b=bs)
            else:
                # print(f'[fuck debug here] {history}') # here !!!
                if (history is None) or (history == 'v2'):
                    latent = (latent + pos_embed).to(latent.dtype)
                latent = rearrange(latent, '(b t) n c -> b t n c', b=bs, t=temp)

        else:
            assert self.pos_embed_type == "rope", "Only supporting the sincos and rope embedding"
            latent = rearrange(latent, '(b t) n c -> b t n c', b=bs, t=temp)
        
        return latent

    def forward(self, latent):
        """
        Arguments:
            past_condition_latents (Torch.FloatTensor): The past latent during the generation
            flatten_input (bool): True indicate flatten the latent into 1D sequence
        """

        if isinstance(latent, list):
            output_list = []
            
            for latent_ in latent:
                if not isinstance(latent_, list):
                    latent_ = [latent_]

                output_latent = []
                time_index = 0
                ori_height, ori_width = latent_[-1].shape[-2:]
                for each_latent in latent_:
                    hidden_state = self.forward_func(each_latent, time_index=time_index, ori_height=ori_height, ori_width=ori_width)
                    time_index  += each_latent.shape[2]
                    hidden_state = rearrange(hidden_state, "b t n c -> b (t n) c")
                    output_latent.append(hidden_state)

                # print(f'[output_latent] {[output_latent_.shape for output_latent_ in output_latent]}')
                output_latent = torch.cat(output_latent, dim=1)
                output_list.append(output_latent)

            return output_list

        else:
            hidden_states = self.forward_func(latent)
            hidden_states = rearrange(hidden_states, "b t n c -> b (t n) c")
            return hidden_states
    

    def forward_history_v1(
        self, 
        history_latent, 
        downsample_num=None,
    ):
        """
            history_latent: [b c t h w]
        """
        ori_height, ori_width = history_latent[-1].shape[-2:]
        hidden_state = self.forward_func(history_latent, ori_height=ori_height, ori_width=ori_width, history='v1')
        hidden_state = rearrange(hidden_state, "b t n c -> b (t n) c")

        if downsample_num is not None:
            b, n = hidden_state.size(0), hidden_state.size(1)
            rand_indices    = torch.rand(b, n).argsort(dim=1)
            sampled_indices = rand_indices[:, :downsample_num]
            hidden_state    = hidden_state[torch.arange(b).unsqueeze(1), sampled_indices]

        return hidden_state


    def forward_history_v2(
        self, 
        history_latent, 
        history_downsample_ratio=1,
    ):
        """
            history_latent: [b c t h w]
        """
        t, ori_height, ori_width = history_latent[-1].shape[-3:]

        
        ori_height //= history_downsample_ratio
        ori_width  //= history_downsample_ratio
        history_latent = rearrange(history_latent, 'b c t h w -> (b t) c h w')
        history_latent = F.interpolate(history_latent, size=(ori_height, ori_width), mode='bilinear')
        history_latent = rearrange(history_latent, '(b t) c h w -> b c t h w', t=t)

        hidden_state = self.forward_func(history_latent, ori_height=ori_height, ori_width=ori_width, history='v2')
        hidden_state = rearrange(hidden_state, "b t n c -> b (t n) c")

        return hidden_state


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class EmbedNDRoPE(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(2)



class SD3TextEncoderWithMask(nn.Module):
    def __init__(self, model_path, torch_dtype):
        super().__init__()
        # CLIP-L
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
        self.tokenizer_max_length = self.tokenizer.model_max_length
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(os.path.join(model_path, 'text_encoder'), torch_dtype=torch_dtype)

        # CLIP-G
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer_2'))
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(os.path.join(model_path, 'text_encoder_2'), torch_dtype=torch_dtype)

        # T5
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(os.path.join(model_path, 'tokenizer_3'))
        self.text_encoder_3 = T5EncoderModel.from_pretrained(os.path.join(model_path, 'text_encoder_3'), torch_dtype=torch_dtype)
    
        self._freeze()

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 128,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)
        prompt_embeds = self.text_encoder_3(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return pooled_prompt_embeds

    def encode_prompt(self, 
        prompt, 
        num_images_per_prompt=1, 
        clip_skip: Optional[int] = None,
        device=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=0,
        )
        pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=1,
        )
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
        )
        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds

    def forward(self, input_prompts, device):
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.encode_prompt(input_prompts, 1, clip_skip=None, device=device)

        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds

class MMDiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        pos_embed_max_size: int = 192,
        max_num_frames: int = 200,
        qk_norm: str = 'rms_norm',
        pos_embed_type: str = 'rope',
        temp_pos_embed_type: str = 'sincos',
        joint_attention_dim: int = 4096,
        use_gradient_checkpointing: bool = False,
        use_flash_attn: bool = True,
        use_temporal_causal: bool = False,
        use_t5_mask: bool = False,
        add_temp_pos_embed: bool = False,
        interp_condition_pos: bool = False,
        gradient_checkpointing_ratio: float = 0.6,
    ):
        super().__init__()

        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        assert temp_pos_embed_type in ['rope', 'sincos']

        # The input latent embeder, using the name pos_embed to remain the same with SD#
        self.pos_embed = PatchEmbed3D(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
            max_num_frames=max_num_frames,
            pos_embed_type=pos_embed_type,
            temp_pos_embed_type=temp_pos_embed_type,
            add_temp_pos_embed=add_temp_pos_embed,
            interp_condition_pos=interp_condition_pos,
        )

        # The RoPE EMbedding
        if pos_embed_type == 'rope':
            self.rope_embed = EmbedNDRoPE(self.inner_dim, 10000, axes_dim=[16, 24, 24])
        else:
            self.rope_embed = None

        if temp_pos_embed_type == 'rope':
            self.temp_rope_embed = EmbedNDRoPE(self.inner_dim, 10000, axes_dim=[attention_head_dim])
        else:
            self.temp_rope_embed = None

        self.time_text_embed = CombinedTimestepConditionEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim,
        )
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=self.inner_dim,
                    qk_norm=qk_norm,
                    context_pre_only=i == num_layers - 1,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.gradient_checkpointing = use_gradient_checkpointing
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio

        self.patch_size = patch_size
        self.use_flash_attn = use_flash_attn
        self.use_temporal_causal = use_temporal_causal
        self.pos_embed_type = pos_embed_type
        self.temp_pos_embed_type = temp_pos_embed_type
        self.add_temp_pos_embed = add_temp_pos_embed

        if self.use_temporal_causal:
            print("Using temporal causal attention")
            assert self.use_flash_attn is False, "The flash attention does not support temporal causal"
        
        if interp_condition_pos:
            print("We interp the position embedding of condition latents")

        # init weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.pos_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pos_embed.proj.bias, 0)

        # Initialize all the conditioning to normal init
        nn.init.normal_(self.time_text_embed.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.time_text_embed.timestep_embedder.linear_2.weight, std=0.02)
        nn.init.normal_(self.time_text_embed.text_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.time_text_embed.text_embedder.linear_2.weight, std=0.02)
        nn.init.normal_(self.context_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)
            nn.init.constant_(block.norm1_context.linear.weight, 0)
            nn.init.constant_(block.norm1_context.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    @torch.no_grad()
    def _prepare_latent_image_ids(self, batch_size, temp, height, width, device):
        latent_image_ids = torch.zeros(temp, height, width, 3)
        latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(temp)[:, None, None]
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[None, :, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, None, :]

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1, 1)
        latent_image_ids = rearrange(latent_image_ids, 'b t h w c -> b (t h w) c')
        return latent_image_ids.to(device=device)

    @torch.no_grad()
    def _prepare_pyramid_latent_image_ids(self, batch_size, temp_list, height_list, width_list, device):
        base_width = width_list[-1]; base_height = height_list[-1]
        assert base_width == max(width_list)
        assert base_height == max(height_list)

        image_ids_list = []
        for temp, height, width in zip(temp_list, height_list, width_list):
            latent_image_ids = torch.zeros(temp, height, width, 3)

            if height != base_height:
                height_pos = F.interpolate(torch.arange(base_height)[None, None, :].float(), height, mode='linear').squeeze(0, 1)
            else:
                height_pos = torch.arange(base_height).float()
            if width != base_width:
                width_pos = F.interpolate(torch.arange(base_width)[None, None, :].float(), width, mode='linear').squeeze(0, 1)
            else:
                width_pos = torch.arange(base_width).float()

            latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(temp)[:, None, None]  
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + height_pos[None, :, None]
            latent_image_ids[..., 2] = latent_image_ids[..., 2] + width_pos[None, None, :]
            latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1, 1)
            latent_image_ids = rearrange(latent_image_ids, 'b t h w c -> b (t h w) c').to(device)
            image_ids_list.append(latent_image_ids)
    
        return image_ids_list

    @torch.no_grad()
    def _prepare_temporal_rope_ids(self, batch_size, temp, height, width, device, start_time_stamp=0):
        latent_image_ids = torch.zeros(temp, height, width, 1)
        latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(start_time_stamp, start_time_stamp + temp)[:, None, None]
        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1, 1)
        latent_image_ids = rearrange(latent_image_ids, 'b t h w c -> b (t h w) c')
        return latent_image_ids.to(device=device)

    @torch.no_grad()
    def _prepare_pyramid_temporal_rope_ids(self, sample, batch_size, device):
        image_ids_list = []

        for i_b, sample_ in enumerate(sample):
            if not isinstance(sample_, list):
                sample_ = [sample_]

            cur_image_ids = []
            start_time_stamp = 0

            for clip_ in sample_:
                _, _, temp, height, width = clip_.shape
                height = height // self.patch_size
                width  = width  // self.patch_size
                cur_image_ids.append(self._prepare_temporal_rope_ids(batch_size, temp, height, width, device, start_time_stamp=start_time_stamp))
                start_time_stamp += temp

            cur_image_ids = torch.cat(cur_image_ids, dim=1)
            image_ids_list.append(cur_image_ids)

        return image_ids_list

    def merge_input(self, sample, encoder_hidden_length, encoder_attention_mask):
        """
            Merge the input video with different resolutions into one sequence
            Sample: From low resolution to high resolution
        """
        if isinstance(sample[0], list):
            device = sample[0][-1].device
            pad_batch_size = sample[0][-1].shape[0]
        else:
            device = sample[0].device
            pad_batch_size = sample[0].shape[0]

        num_stages = len(sample)
        height_list = []
        width_list  = []
        temp_list   = []
        trainable_token_list = []

        for i_b, sample_ in enumerate(sample):
            if isinstance(sample_, list):
                sample_ = sample_[-1]
            _, _, temp, height, width = sample_.shape
            height = height // self.patch_size
            width  = width  // self.patch_size
            temp_list.append(temp)
            height_list.append(height)
            width_list.append(width)
            trainable_token_list.append(height * width * temp)

        # prepare the RoPE embedding if needed
        if self.pos_embed_type == 'rope':
            # TODO: support the 3D Rope for video
            raise NotImplementedError("Not compatible with video generation now")
            text_ids = torch.zeros(pad_batch_size, encoder_hidden_length, 3).to(device=device)
            image_ids_list = self._prepare_pyramid_latent_image_ids(pad_batch_size, temp_list, height_list, width_list, device)
            input_ids_list = [torch.cat([text_ids, image_ids], dim=1) for image_ids in image_ids_list]
            image_rotary_emb = [self.rope_embed(input_ids) for input_ids in input_ids_list]  # [bs, seq_len, 1, head_dim // 2, 2, 2]
        else:
            if self.temp_pos_embed_type == 'rope' and self.add_temp_pos_embed:
                image_ids_list = self._prepare_pyramid_temporal_rope_ids(sample, pad_batch_size, device)
                text_ids = torch.zeros(pad_batch_size, encoder_attention_mask.shape[1], 1).to(device=device)    
                input_ids_list = [torch.cat([text_ids, image_ids], dim=1) for image_ids in image_ids_list]
                image_rotary_emb = [self.temp_rope_embed(input_ids) for input_ids in input_ids_list]  # [bs, seq_len, 1, head_dim // 2, 2, 2]

                # if is_sequence_parallel_initialized():
                #     sp_group = get_sequence_parallel_group()
                #     sp_group_size = get_sequence_parallel_world_size()
                #     concat_output = True if self.training else False
                #     image_rotary_emb = [all_to_all(x_.repeat(1, 1, sp_group_size, 1, 1, 1), sp_group, sp_group_size, scatter_dim=2, gather_dim=0, concat_output=concat_output) for x_ in image_rotary_emb]
                #     input_ids_list = [all_to_all(input_ids.repeat(1, 1, sp_group_size), sp_group, sp_group_size, scatter_dim=2, gather_dim=0, concat_output=concat_output) for input_ids in input_ids_list]
            else:
                image_rotary_emb = None

        hidden_states = self.pos_embed(sample)  # hidden states is a list of [b c t h w] b = real_b // num_stages

        hidden_length = []

        for i_b in range(num_stages):
            hidden_length.append(hidden_states[i_b].shape[1])
        # print(f'[hidden_length] {hidden_length}')

        # prepare the attention mask
        if self.use_flash_attn:
            attention_mask = None
            indices_list = []
            for i_p, length in enumerate(hidden_length):
                pad_attention_mask = torch.ones((pad_batch_size, length), dtype=encoder_attention_mask.dtype).to(device)
                pad_attention_mask = torch.cat([encoder_attention_mask[i_p::num_stages], pad_attention_mask], dim=1)
                
                if is_sequence_parallel_initialized():
                    sp_group = get_sequence_parallel_group()
                    sp_group_size = get_sequence_parallel_world_size()
                    pad_attention_mask = all_to_all(pad_attention_mask.unsqueeze(2).repeat(1, 1, sp_group_size), sp_group, sp_group_size, scatter_dim=2, gather_dim=0)
                    pad_attention_mask = pad_attention_mask.squeeze(2)

                seqlens_in_batch = pad_attention_mask.sum(dim=-1, dtype=torch.int32)
                indices = torch.nonzero(pad_attention_mask.flatten(), as_tuple=False).flatten()

                indices_list.append(
                    {
                        'indices': indices,
                        'seqlens_in_batch': seqlens_in_batch,
                    }
                )
            encoder_attention_mask = indices_list

        else:
            assert encoder_attention_mask.shape[1] == encoder_hidden_length
            real_batch_size = encoder_attention_mask.shape[0]
            # prepare text ids
            text_ids = torch.arange(1, real_batch_size + 1, dtype=encoder_attention_mask.dtype).unsqueeze(1).repeat(1, encoder_hidden_length)
            text_ids = text_ids.to(device)
            text_ids[encoder_attention_mask == 0] = 0

            # prepare image ids
            image_ids = torch.arange(1, real_batch_size + 1, dtype=encoder_attention_mask.dtype).unsqueeze(1).repeat(1, max(hidden_length))
            image_ids = image_ids.to(device)
            image_ids_list = []
            for i_p, length in enumerate(hidden_length):
                image_ids_list.append(image_ids[i_p::num_stages][:, :length])

            if is_sequence_parallel_initialized():
                sp_group = get_sequence_parallel_group()
                sp_group_size = get_sequence_parallel_world_size()
                concat_output = True if self.training else False
                text_ids = all_to_all(text_ids.unsqueeze(2).repeat(1, 1, sp_group_size), sp_group, sp_group_size, scatter_dim=2, gather_dim=0, concat_output=concat_output).squeeze(2)
                image_ids_list = [all_to_all(image_ids_.unsqueeze(2).repeat(1, 1, sp_group_size), sp_group, sp_group_size, scatter_dim=2, gather_dim=0, concat_output=concat_output).squeeze(2) for image_ids_ in image_ids_list]

            attention_mask = []
            for i_p in range(len(hidden_length)):
                image_ids = image_ids_list[i_p]
                token_ids = torch.cat([text_ids[i_p::num_stages], image_ids], dim=1)
                stage_attention_mask = rearrange(token_ids, 'b i -> b 1 i 1') == rearrange(token_ids, 'b j -> b 1 1 j')  # [bs, 1, q_len, k_len]
                if self.use_temporal_causal:
                    input_order_ids = input_ids_list[i_p].squeeze(2)
                    temporal_causal_mask = rearrange(input_order_ids, 'b i -> b 1 i 1') >= rearrange(input_order_ids, 'b j -> b 1 1 j')
                    stage_attention_mask = stage_attention_mask & temporal_causal_mask
                attention_mask.append(stage_attention_mask)

        return hidden_states, hidden_length, temp_list, height_list, width_list, trainable_token_list, encoder_attention_mask, attention_mask, image_rotary_emb

    def split_output(self, batch_hidden_states, hidden_length, temps, heights, widths, trainable_token_list):
        # To split the hidden states
        batch_size = batch_hidden_states.shape[0]
        output_hidden_list = []
        batch_hidden_states = torch.split(batch_hidden_states, hidden_length, dim=1)

        if is_sequence_parallel_initialized():
            sp_group_size = get_sequence_parallel_world_size()
            if self.training:
                batch_size = batch_size // sp_group_size

        for i_p, length in enumerate(hidden_length):
            width, height, temp = widths[i_p], heights[i_p], temps[i_p]
            trainable_token_num = trainable_token_list[i_p]
            hidden_states = batch_hidden_states[i_p]

            if is_sequence_parallel_initialized():
                sp_group = get_sequence_parallel_group()
                sp_group_size = get_sequence_parallel_world_size()

                if not self.training:
                    hidden_states = hidden_states.repeat(sp_group_size, 1, 1)

                hidden_states = all_to_all(hidden_states, sp_group, sp_group_size, scatter_dim=0, gather_dim=1)

            # only the trainable token are taking part in loss computation
            hidden_states = hidden_states[:, -trainable_token_num:]

            # unpatchify
            hidden_states = hidden_states.reshape(
                shape=(batch_size, temp, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = rearrange(hidden_states, "b t h w p1 p2 c -> b t (h p1) (w p2) c")
            hidden_states = rearrange(hidden_states, "b t h w c -> b c t h w")
            output_hidden_list.append(hidden_states)

        return output_hidden_list
    
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


    def forward(
        self,
        sample: torch.FloatTensor = None, # [num_stages]
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep_ratio: torch.FloatTensor = None,

        history: torch.FloatTensor = None,
        p_uncond_historyimage: float = None,
        firstframe_list: torch.FloatTensor = None,
        history_mask = None,
        history_version: int = 1,

        # v1
        history_downsample_num: int = None,

        # v2
        history_downsample_ratio: int = None,

        # hidden_states: torch.FloatTensor = None,
        # timestep: torch.FloatTensor = None,
    ):
        # print(f'[TimestepEmbedding info] {self.time_text_embed.timestep_embedder.linear_1.weight.shape, self.time_text_embed.timestep_embedder.linear_1.weight.dtype, self.time_text_embed.timestep_embedder.linear_1.weight.device}')
        # timestep_ratio = timestep
        # Get the timestep embedding
        temb = self.time_text_embed(timestep_ratio, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        # encoder_hidden_length = encoder_hidden_states.shape[1]

        if history is not None:
            
            if   history_version == 1:
                history_hidden_states  = self.pos_embed.forward_history_v1(history, downsample_num=history_downsample_num) # [b n c]
            elif history_version == 2:
                history_hidden_states  = self.pos_embed.forward_history_v2(history, history_downsample_ratio=history_downsample_ratio)

            if history_mask == None: # train phrase
                history_mask = torch.ones(history_hidden_states.shape[:2]).to(encoder_attention_mask)

                bs                                  = encoder_hidden_states.shape[0]
                uncond_mask                         = torch.rand(bs) < p_uncond_historyimage # 1 for no history guidance
                if firstframe_list is not None:
                    uncond_mask = uncond_mask + firstframe_list.to(uncond_mask)
                history_mask[uncond_mask]          *= 0
                history_hidden_states[uncond_mask] *= 0

            encoder_hidden_states  = torch.cat([history_hidden_states.to(encoder_hidden_states), encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat([        history_mask.to(encoder_attention_mask), encoder_attention_mask], dim=1)

            # print(f'[cnm debug] {encoder_hidden_states.shape, history_mask.shape, encoder_attention_mask.shape, uncond_mask, firstframe_list}')

        encoder_hidden_length  = encoder_hidden_states.shape[1]


        # Get the input sequence
        hidden_states, hidden_length, temps, heights, widths, trainable_token_list, encoder_attention_mask, \
                attention_mask, image_rotary_emb = self.merge_input(sample, encoder_hidden_length, encoder_attention_mask)

        # for i_p, hidden_states_ in enumerate(hidden_states):
        #     print(f'[hidden_states_ {i_p}] {hidden_states_.shape}')
        
        # split the long latents if necessary
        if is_sequence_parallel_initialized():
            sp_group = get_sequence_parallel_group()
            sp_group_size = get_sequence_parallel_world_size()
            concat_output = True if self.training else False
            
            # sync the input hidden states
            batch_hidden_states = []
            for i_p, hidden_states_ in enumerate(hidden_states):
                # print(f'[hidden_states_] {hidden_states_.shape, sp_group_size}')
                assert hidden_states_.shape[1] % sp_group_size == 0, f"The sequence length should be divided by sequence parallel size, hidden_states_.shape[1] = {hidden_states_.shape[1]} and sp_group_size = {sp_group_size}."
                hidden_states_ = all_to_all(hidden_states_, sp_group, sp_group_size, scatter_dim=1, gather_dim=0, concat_output=concat_output)
                hidden_length[i_p] = hidden_length[i_p] // sp_group_size
                batch_hidden_states.append(hidden_states_)

            # sync the encoder hidden states
            hidden_states = torch.cat(batch_hidden_states, dim=1)
            encoder_hidden_states = all_to_all(encoder_hidden_states, sp_group, sp_group_size, scatter_dim=1, gather_dim=0, concat_output=concat_output)
            temb = all_to_all(temb.unsqueeze(1).repeat(1, sp_group_size, 1), sp_group, sp_group_size, scatter_dim=1, gather_dim=0, concat_output=concat_output)
            temb = temb.squeeze(1)
        else:
            hidden_states = torch.cat(hidden_states, dim=1)

        # print(hidden_length)
        for i_b, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing and (i_b >= int(len(self.transformer_blocks) * self.gradient_checkpointing_ratio)):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    temb,
                    attention_mask,
                    hidden_length,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, 
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                    attention_mask=attention_mask,
                    hidden_length=hidden_length,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = self.norm_out(hidden_states, temb, hidden_length=hidden_length)

        # print(f'[debug info] hidden_states before {hidden_states.shape}')
        hidden_states = self.proj_out(hidden_states)
        # print(f'[debug info] hidden_states after  {hidden_states.shape}')

        output = self.split_output(hidden_states, hidden_length, temps, heights, widths, trainable_token_list)

        return output
