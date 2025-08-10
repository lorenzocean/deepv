import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from collections import deque, namedtuple
from typing import Tuple, Union, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.activations import get_activation
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.modeling_outputs import AutoencoderKLOutput


def is_context_parallel_initialized():
    # In a non-distributed environment, this will always be false.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return True
    return False

def get_context_parallel_group():
    # Placeholder
    if is_context_parallel_initialized():
        return torch.distributed.group.WORLD
    return None

def get_context_parallel_world_size():
    # Placeholder
    if is_context_parallel_initialized():
        return torch.distributed.get_world_size()
    return 1

def get_context_parallel_rank():
    # Placeholder
    if is_context_parallel_initialized():
        return torch.distributed.get_rank()
    return 0

def get_context_parallel_group_rank():
    # Placeholder
    if is_context_parallel_initialized():
        return torch.distributed.get_rank()
    return 0

def _conv_split(input_, dim=2, kernel_size=1):
    cp_world_size = get_context_parallel_world_size()
    if cp_world_size == 1:
        return input_
    cp_rank = get_context_parallel_rank()
    dim_size = (input_.size()[dim] - kernel_size) // cp_world_size
    if cp_rank == 0:
        output = input_.transpose(dim, 0)[: dim_size + kernel_size].transpose(dim, 0)
    else:
        output = input_.transpose(dim, 0)[
            cp_rank * dim_size + kernel_size : (cp_rank + 1) * dim_size + kernel_size
        ].transpose(dim, 0)
    return output.contiguous()

def _conv_gather(input_, dim=2, kernel_size=1):
    cp_world_size = get_context_parallel_world_size()
    if cp_world_size == 1:
        return input_
    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    input_first_kernel_ = input_.transpose(0, dim)[:kernel_size].transpose(0, dim).contiguous()
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[kernel_size:].transpose(0, dim).contiguous()
    else:
        input_ = input_.transpose(0, dim)[max(kernel_size - 1, 0) :].transpose(0, dim).contiguous()
    tensor_list = [torch.empty_like(torch.cat([input_first_kernel_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]
    if cp_rank == 0:
        input_ = torch.cat([input_first_kernel_, input_], dim=dim)
    tensor_list[cp_rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output

def _cp_pass_from_previous_rank(input_, dim, kernel_size):
    if kernel_size == 1:
        return input_
    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_world_size = get_context_parallel_world_size()
    global_rank = torch.distributed.get_rank()
    input_ = input_.transpose(0, dim)
    send_rank = global_rank + 1
    recv_rank = global_rank - 1
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank += cp_world_size
    recv_buffer = torch.empty_like(input_[-kernel_size + 1 :]).contiguous()
    if cp_rank < cp_world_size - 1:
        req_send = torch.distributed.isend(input_[-kernel_size + 1 :].contiguous(), send_rank, group=group)
    if cp_rank > 0:
        req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)
    if cp_rank == 0:
        input_ = torch.cat([torch.zeros_like(input_[:1])] * (kernel_size - 1) + [input_], dim=0)
    else:
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)
    return input_.transpose(0, dim).contiguous()

def _drop_from_previous_rank(input_, dim, kernel_size):
    return input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim)

class _ConvolutionScatterToContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_split(input_, dim, kernel_size)
    @staticmethod
    def backward(ctx, grad_output):
        return _conv_gather(grad_output, ctx.dim, ctx.kernel_size), None, None

class _ConvolutionGatherFromContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_gather(input_, dim, kernel_size)
    @staticmethod
    def backward(ctx, grad_output):
        return _conv_split(grad_output, ctx.dim, ctx.kernel_size), None, None

class _CPConvolutionPassFromPreviousRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _cp_pass_from_previous_rank(input_, dim, kernel_size)
    @staticmethod
    def backward(ctx, grad_output):
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None

def conv_scatter_to_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionScatterToContextParallelRegion.apply(input_, dim, kernel_size)

def conv_gather_from_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionGatherFromContextParallelRegion.apply(input_, dim, kernel_size)

def cp_pass_from_previous_rank(input_, dim, kernel_size):
    return _CPConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size)

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def is_odd(n):
    return (n % 2) != 0

class CausalGroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = super().forward(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x

class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        pad_mode: str ='constant',
        **kwargs
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = cast_tuple(kernel_size, 3)
    
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.time_kernel_size = time_kernel_size
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        dilation = kwargs.pop('dilation', 1)
        self.pad_mode = pad_mode

        if isinstance(stride, int):
            stride = (stride, 1, 1)
    
        time_pad = dilation * (time_kernel_size - 1)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.temporal_stride = stride[0]
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        self.time_uncausal_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
        self.cache_front_feat = deque()

    def _clear_context_parallel_cache(self):
        del self.cache_front_feat
        self.cache_front_feat = deque()

    def context_parallel_forward(self, x):
        cp_rank = get_context_parallel_rank()
        if self.time_kernel_size == 3 and ((cp_rank == 0 and x.shape[2] <= 2) or (cp_rank != 0 and x.shape[2] <= 1)):
            x = cp_pass_from_previous_rank(x, dim=2, kernel_size=2)
            trans_x = cp_pass_from_previous_rank(x[:, :, :-1], dim=2, kernel_size=2)
            x = torch.cat([trans_x, x[:, :,-1:]], dim=2)
        else:
            x = cp_pass_from_previous_rank(x, dim=2, kernel_size=self.time_kernel_size)
        
        x = F.pad(x, self.time_uncausal_padding, mode='constant')

        if cp_rank != 0:
            if self.temporal_stride == 2 and self.time_kernel_size == 3:
                x = x[:,:,1:]
        x = self.conv(x)
        return x

    def forward(self, x, is_init_image=True, temporal_chunk=False):
        if is_context_parallel_initialized():
            return self.context_parallel_forward(x)
        
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        if not temporal_chunk:
            x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        else:
            assert not self.training, "The feature cache should not be used in training"
            if is_init_image:
                x = F.pad(x, self.time_causal_padding, mode=pad_mode)
                self._clear_context_parallel_cache()
                self.cache_front_feat.append(x[:, :, -2:].clone().detach())
            else:
                x = F.pad(x, self.time_uncausal_padding, mode=pad_mode)
                video_front_context = self.cache_front_feat.pop()
                self._clear_context_parallel_cache()

                if self.temporal_stride == 1 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context, x], dim=2)
                elif self.temporal_stride == 2 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context[:,:,-1:], x], dim=2)

                self.cache_front_feat.append(x[:, :, -2:].clone().detach())
        
        x = self.conv(x)
        return x

class CausalResnetBlock3D(nn.Module):
    def __init__(
        self, *, in_channels: int, out_channels: Optional[int] = None,
        conv_shortcut: bool = False, dropout: float = 0.0,
        temb_channels: int = 512, groups: int = 32,
        groups_out: Optional[int] = None, pre_norm: bool = True,
        eps: float = 1e-6, non_linearity: str = "swish",
        time_embedding_norm: str = "default", output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None, conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        self.norm1 = CausalGroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)
        self.norm2 = CausalGroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = CausalConv3d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1)
        self.nonlinearity = get_activation(non_linearity)
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(
                in_channels, conv_2d_out_channels, kernel_size=1,
                stride=1, bias=conv_shortcut_bias,
            )

    def forward(
        self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor = None,
        is_init_image=True, temporal_chunk=False,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        if temb is not None:
             hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor

class CausalDownsample2x(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool = True,
        out_channels: Optional[int] = None, name: str = "conv",
        kernel_size=3, bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = (1, 2, 2)
        self.name = name
        if use_conv:
            self.conv = CausalConv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, hidden_states: torch.FloatTensor, is_init_image=True, temporal_chunk=False) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        return hidden_states

class CausalTemporalDownsample2x(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool = True,
        out_channels: Optional[int] = None, kernel_size=3, bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = (2, 1, 1)
        if use_conv:
            self.conv = CausalConv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, bias=bias
            )
        else:
            raise NotImplementedError("Not implemented for temporal downsample without conv")

    def forward(self, hidden_states: torch.FloatTensor, is_init_image=True, temporal_chunk=False) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        return hidden_states

class CausalUpsample2x(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool = True,
        out_channels: Optional[int] = None, name: str = "conv",
        kernel_size: Optional[int] = 3, bias=True, interpolate=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name
        self.interpolate = interpolate
        if interpolate:
            raise NotImplementedError("Not implemented for spatial upsample with interpolate")
        else:
            self.conv = CausalConv3d(self.channels, self.out_channels * 4, kernel_size=kernel_size, stride=1, bias=bias)

    def forward(
        self, hidden_states: torch.FloatTensor,
        is_init_image=True, temporal_chunk=False,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        hidden_states = rearrange(hidden_states, 'b (c p1 p2) t h w -> b c t (h p1) (w p2)', p1=2, p2=2)
        return hidden_states

class CausalTemporalUpsample2x(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool = True,
        out_channels: Optional[int] = None, kernel_size: Optional[int] = 3,
        bias=True, interpolate=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.interpolate = interpolate
        if interpolate:
            raise NotImplementedError("Not implemented for spatial upsample with interpolate")
        else:
            self.conv = CausalConv3d(self.channels, self.out_channels * 2, kernel_size=kernel_size, stride=1, bias=bias)

    def forward(
        self, hidden_states: torch.FloatTensor,
        is_init_image=True, temporal_chunk=False,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        hidden_states = rearrange(hidden_states, 'b (c p) t h w -> b c (t p) h w', p=2)
        if is_init_image:
            hidden_states = hidden_states[:, :, 1:]
        return hidden_states

class CausalUNetMidBlock2D(nn.Module):
    def __init__(
        self, in_channels: int, temb_channels: int, dropout: float = 0.0,
        num_layers: int = 1, resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish",
        resnet_groups: int = 32, attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True, add_attention: bool = True,
        attention_head_dim: int = 1, output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention
        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        resnets = [
            CausalResnetBlock3D(
                in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels,
                eps=resnet_eps, groups=resnet_groups, dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels, heads=in_channels // attention_head_dim, dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)
            resnets.append(
                CausalResnetBlock3D(
                    in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels,
                    eps=resnet_eps, groups=resnet_groups, dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None,
            is_init_image=True, temporal_chunk=False) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        t = hidden_states.shape[2]
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, 'b c t h w -> (b t) c h w')
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(hidden_states, '(b t) c h w -> b c t h w', t=t)
            hidden_states = resnet(hidden_states, temb, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        return hidden_states

class DownEncoderBlockCausal3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0,
        num_layers: int = 1, resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish",
        resnet_groups: int = 32, resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0, add_spatial_downsample: bool = True,
        add_temporal_downsample: bool = False, downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                CausalResnetBlock3D(
                    in_channels=in_channels, out_channels=out_channels, temb_channels=None,
                    eps=resnet_eps, groups=resnet_groups, dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_spatial_downsample:
            self.downsamplers = nn.ModuleList([
                CausalDownsample2x(out_channels, use_conv=True, out_channels=out_channels)
            ])
        else:
            self.downsamplers = None
        if add_temporal_downsample:
            self.temporal_downsamplers = nn.ModuleList([
                CausalTemporalDownsample2x(out_channels, use_conv=True, out_channels=out_channels)
            ])
        else:
            self.temporal_downsamplers = None

    def forward(self, hidden_states: torch.FloatTensor, is_init_image=True, temporal_chunk=False) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        if self.temporal_downsamplers is not None:
            for temporal_downsampler in self.temporal_downsamplers:
                hidden_states = temporal_downsampler(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        return hidden_states

class UpDecoderBlockCausal3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, resolution_idx: Optional[int] = None,
        dropout: float = 0.0, num_layers: int = 1, resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish",
        resnet_groups: int = 32, resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0, add_spatial_upsample: bool = True,
        add_temporal_upsample: bool = False, temb_channels: Optional[int] = None,
        interpolate: bool = True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                CausalResnetBlock3D(
                    in_channels=input_channels, out_channels=out_channels, temb_channels=temb_channels,
                    eps=resnet_eps, groups=resnet_groups, dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_spatial_upsample:
            self.upsamplers = nn.ModuleList([CausalUpsample2x(out_channels, use_conv=True, out_channels=out_channels, interpolate=interpolate)])
        else:
            self.upsamplers = None
        if add_temporal_upsample:
            self.temporal_upsamplers = nn.ModuleList([CausalTemporalUpsample2x(out_channels, use_conv=True, out_channels=out_channels, interpolate=interpolate)])
        else:
            self.temporal_upsamplers = None

    def forward(
        self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None,
        is_init_image=True, temporal_chunk=False,
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        if self.temporal_upsamplers is not None:
            for temporal_upsampler in self.temporal_upsamplers:
                hidden_states = temporal_upsampler(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        return hidden_states

def get_down_block(
    down_block_type: str, num_layers: int, in_channels: int,
    out_channels: int, temb_channels: int, add_spatial_downsample: bool,
    add_temporal_downsample: bool, resnet_eps: float, resnet_act_fn: str,
    resnet_groups: int, attention_head_dim: int, resnet_time_scale_shift: str,
    dropout: float,
):
    if down_block_type == "DownEncoderBlockCausal3D":
        return DownEncoderBlockCausal3D(
            num_layers=num_layers, in_channels=in_channels, out_channels=out_channels,
            add_spatial_downsample=add_spatial_downsample, add_temporal_downsample=add_temporal_downsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            dropout=dropout,
        )
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type: str, num_layers: int, in_channels: int,
    out_channels: int, prev_output_channel: int, temb_channels: int,
    add_spatial_upsample: bool, add_temporal_upsample: bool,
    resnet_eps: float, resnet_act_fn: str, resnet_groups: int,
    attention_head_dim: int, resnet_time_scale_shift: str,
    dropout: float, interpolate: bool,
):
    if up_block_type == "UpDecoderBlockCausal3D":
        return UpDecoderBlockCausal3D(
            num_layers=num_layers, in_channels=in_channels, out_channels=out_channels,
            add_spatial_upsample=add_spatial_upsample, add_temporal_upsample=add_temporal_upsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            dropout=dropout, temb_channels=temb_channels, interpolate=interpolate,
        )
    raise ValueError(f"{up_block_type} does not exist.")

@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        sample = randn_tensor(self.mean.shape, generator=generator, device=self.parameters.device, dtype=self.parameters.dtype)
        return self.mean + self.std * sample

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device)
        if other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3, 4],
            )
    def mode(self) -> torch.Tensor:
        return self.mean

class CausalVaeEncoder(nn.Module):
    def __init__(
        self, in_channels: int = 3, out_channels: int = 4,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlockCausal3D",),
        block_out_channels: Tuple[int, ...] = (128,), layers_per_block: Tuple[int, ...] = (2,),
        act_fn: str = "silu", norm_num_groups: int = 32, double_z: bool = True,
        spatial_down_sample: Tuple[bool, ...] = (True,), temporal_down_sample: Tuple[bool, ...] = (False,),
        block_dropout: Tuple[float, ...] = (0.0,), mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1)
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            down_block = get_down_block(
                down_block_type, num_layers=layers_per_block[i], in_channels=input_channel,
                out_channels=output_channel, temb_channels=None, add_spatial_downsample=spatial_down_sample[i],
                add_temporal_downsample=temporal_down_sample[i], resnet_eps=1e-6, resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups, attention_head_dim=output_channel,
                resnet_time_scale_shift="default", dropout=block_dropout[i],
            )
            self.down_blocks.append(down_block)

        self.mid_block = CausalUNetMidBlock2D(
            in_channels=block_out_channels[-1], resnet_eps=1e-6, resnet_act_fn=act_fn,
            output_scale_factor=1.0, resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1], resnet_groups=norm_num_groups,
            temb_channels=None, add_attention=mid_block_add_attention, dropout=block_dropout[-1],
        )

        self.conv_norm_out = CausalGroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3, stride=1)
        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor, is_init_image=True, temporal_chunk=False) -> torch.FloatTensor:
        sample = self.conv_in(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        if self.training and self.gradient_checkpointing:
            # Custom forward for checkpointing
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            # Down blocks
            for down_block in self.down_blocks:
                sample = checkpoint(create_custom_forward(down_block), sample, is_init_image, temporal_chunk)
            # Mid block
            sample = checkpoint(create_custom_forward(self.mid_block), sample, is_init_image, temporal_chunk)
        else:
            for down_block in self.down_blocks:
                sample = down_block(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
            sample = self.mid_block(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        return sample

class CausalVaeDecoder(nn.Module):
    def __init__(
        self, in_channels: int = 4, out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlockCausal3D",),
        block_out_channels: Tuple[int, ...] = (512,), layers_per_block: Tuple[int, ...] = (2,),
        act_fn: str = "silu", norm_num_groups: int = 32, mid_block_add_attention: bool = True,
        spatial_up_sample: Tuple[bool, ...] = (True,), temporal_up_sample: Tuple[bool, ...] = (False,),
        block_dropout: Tuple[float, ...] = (0.0,), interpolate: bool = True,
    ):
        super().__init__()
        self.conv_in = CausalConv3d(in_channels, block_out_channels[-1], kernel_size=3, stride=1)
        self.mid_block = CausalUNetMidBlock2D(
            in_channels=block_out_channels[-1], resnet_eps=1e-6, resnet_act_fn=act_fn,
            output_scale_factor=1.0, resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1], resnet_groups=norm_num_groups,
            temb_channels=None, add_attention=mid_block_add_attention, dropout=block_dropout[-1],
        )

        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            up_block = get_up_block(
                up_block_type, num_layers=layers_per_block[i], in_channels=prev_output_channel,
                out_channels=output_channel, prev_output_channel=None, temb_channels=None,
                add_spatial_upsample=spatial_up_sample[i], add_temporal_upsample=temporal_up_sample[i],
                resnet_eps=1e-6, resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
                attention_head_dim=output_channel, resnet_time_scale_shift="default",
                dropout=block_dropout[i], interpolate=interpolate,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = CausalGroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[0], out_channels, kernel_size=3, stride=1)
        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor, is_init_image=True, temporal_chunk=False) -> torch.FloatTensor:
        sample = self.conv_in(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            # Mid block
            sample = checkpoint(create_custom_forward(self.mid_block), sample, is_init_image, temporal_chunk)
            # Up blocks
            for up_block in self.up_blocks:
                sample = checkpoint(create_custom_forward(up_block), sample, is_init_image, temporal_chunk)
        else:
            sample = self.mid_block(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
            for up_block in self.up_blocks:
                sample = up_block(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        return sample

class CausalVideoVAE(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # encoder related
        encoder_in_channels: int = 3,
        encoder_out_channels: int = 4,
        encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
        encoder_down_block_types: Tuple[str, ...] = (
            "DownEncoderBlockCausal3D", "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D", "DownEncoderBlockCausal3D",
        ),
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        encoder_spatial_down_sample: Tuple[bool, ...] = (True, True, True, False),
        encoder_temporal_down_sample: Tuple[bool, ...] = (False, False, False, False),
        encoder_block_dropout: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
        encoder_act_fn: str = "silu",
        encoder_norm_num_groups: int = 32,
        # decoder related
        decoder_in_channels: int = 4,
        decoder_out_channels: int = 3,
        decoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
        decoder_up_block_types: Tuple[str, ...] = (
            "UpDecoderBlockCausal3D", "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D", "UpDecoderBlockCausal3D",
        ),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decoder_spatial_up_sample: Tuple[bool, ...] = (True, True, True, False),
        decoder_temporal_up_sample: Tuple[bool, ...] = (False, False, False, False),
        decoder_block_dropout: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
        decoder_act_fn: str = "silu",
        decoder_norm_num_groups: int = 32,
        # general
        scaling_factor: float = 0.18215,
        sample_size: int = 256,
        downsample_scale: int = 8,
        interpolate: bool = True,

    ):
        super().__init__()

        self.encoder = CausalVaeEncoder(
            in_channels=encoder_in_channels,
            out_channels=encoder_out_channels,
            down_block_types=encoder_down_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            act_fn=encoder_act_fn,
            norm_num_groups=encoder_norm_num_groups,
            double_z=True,
            spatial_down_sample=encoder_spatial_down_sample,
            temporal_down_sample=encoder_temporal_down_sample,
            block_dropout=encoder_block_dropout,
        )

        self.decoder = CausalVaeDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            up_block_types=decoder_up_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            act_fn=decoder_act_fn,
            norm_num_groups=decoder_norm_num_groups,
            spatial_up_sample=decoder_spatial_up_sample,
            temporal_up_sample=decoder_temporal_up_sample,
            block_dropout=decoder_block_dropout,
            interpolate=interpolate
        )
        self.quant_conv = CausalConv3d(2 * encoder_out_channels, 2 * encoder_out_channels, 1)
        self.post_quant_conv = CausalConv3d(encoder_out_channels, decoder_in_channels, 1)
        self.downsample_scale = downsample_scale
        
        self.use_tiling = False
        self.tile_sample_min_size = self.config.sample_size
        self.tile_latent_min_size = int(self.tile_sample_min_size / self.downsample_scale)
        self.encode_tile_overlap_factor = 0.25
        self.decode_tile_overlap_factor = 0.25
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True,
        is_init_image=True, temporal_chunk=False, window_size=16, tile_sample_min_size=256
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:

        self.tile_sample_min_size = tile_sample_min_size
        self.tile_latent_min_size = int(tile_sample_min_size / self.downsample_scale)
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict, is_init_image=is_init_image, temporal_chunk=temporal_chunk, window_size=window_size)

        if temporal_chunk:
            moments = self.chunk_encode(x, window_size=window_size)
        else:
            h = self.encoder(x, is_init_image=is_init_image, temporal_chunk=False)
            moments = self.quant_conv(h, is_init_image=is_init_image, temporal_chunk=False)
            
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @torch.no_grad()
    def chunk_encode(self, x: torch.FloatTensor, window_size=16):
        num_frames = x.shape[2]
        init_window_size = window_size + 1
        frame_list = [x[:,:,:init_window_size]]
        full_chunk_size = (num_frames - init_window_size) // window_size
        fid = init_window_size
        for idx in range(full_chunk_size):
            frame_list.append(x[:, :, fid:fid+window_size])
            fid += window_size
        if fid < num_frames:
            frame_list.append(x[:, :, fid:])
        latent_list = []
        for idx, frames in enumerate(frame_list):
            is_init = (idx == 0)
            h = self.encoder(frames, is_init_image=is_init, temporal_chunk=True)
            moments = self.quant_conv(h, is_init_image=is_init, temporal_chunk=True)
            latent_list.append(moments)
        return torch.cat(latent_list, dim=2)

    def decode(self, z: torch.FloatTensor, return_dict: bool = True, is_init_image=True, 
                 temporal_chunk=False, window_size=2, tile_sample_min_size=256) -> Union[DecoderOutput, torch.FloatTensor]:

        self.tile_sample_min_size = tile_sample_min_size
        self.tile_latent_min_size = int(tile_sample_min_size / self.downsample_scale)
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict, is_init_image=is_init_image, temporal_chunk=temporal_chunk, window_size=window_size)
        
        if temporal_chunk:
            dec = self.chunk_decode(z, window_size=window_size)
        else:
            z = self.post_quant_conv(z, is_init_image=is_init_image, temporal_chunk=False)
            dec = self.decoder(z, is_init_image=is_init_image, temporal_chunk=False)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    @torch.no_grad()
    def chunk_decode(self, z: torch.FloatTensor, window_size=2):
        num_frames = z.shape[2]
        init_window_size = window_size + 1
        frame_list = [z[:,:,:init_window_size]]
        full_chunk_size = (num_frames - init_window_size) // window_size
        fid = init_window_size
        for idx in range(full_chunk_size):
            frame_list.append(z[:, :, fid:fid+window_size])
            fid += window_size
        if fid < num_frames:
            frame_list.append(z[:, :, fid:])
        dec_list = []
        for idx, frames in enumerate(frame_list):
            is_init = (idx == 0)
            z_h = self.post_quant_conv(frames, is_init_image=is_init, temporal_chunk=True)
            dec = self.decoder(z_h, is_init_image=is_init, temporal_chunk=True)
            dec_list.append(dec)
        return torch.cat(dec_list, dim=2)
        
    def forward(
        self, sample: torch.FloatTensor, sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None, is_init_image=True, temporal_chunk=False
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        
        posterior = self.encode(sample, return_dict=False, is_init_image=is_init_image, temporal_chunk=temporal_chunk)[0]
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
            
        dec = self.decode(z, return_dict=False, is_init_image=is_init_image, temporal_chunk=temporal_chunk)[0]
        return dec, posterior
        
    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b
        
    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True, is_init_image=True, temporal_chunk=False, window_size=16) -> AutoencoderKLOutput:
        overlap_size = int(self.tile_sample_min_size * (1 - self.encode_tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.encode_tile_overlap_factor)
        row_limit    = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                if temporal_chunk:
                    tile = self.chunk_encode(tile, window_size=window_size)
                else:
                    tile = self.encoder(tile, is_init_image=True, temporal_chunk=False)
                    tile = self.quant_conv(tile, is_init_image=True, temporal_chunk=False)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))
        moments = torch.cat(result_rows, dim=3)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True, is_init_image=True, temporal_chunk=False, window_size=2) -> Union[DecoderOutput, torch.FloatTensor]:
        overlap_size = int(self.tile_latent_min_size * (1 - self.decode_tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.decode_tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decode(tile, return_dict=False, is_init_image=is_init_image, temporal_chunk=temporal_chunk, window_size=window_size)[0]
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))
        dec = torch.cat(result_rows, dim=3)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
