# ------------------------------------------------------------------------------------------------
# Deformable DETR + SEG-conditioned TCPD-v1/v2 residuals
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 128
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # TCPD-v1 global (target-level broadcast)
        self.tcpd_gate_offset = nn.Parameter(torch.zeros(1))
        self.tcpd_gate_attn = nn.Parameter(torch.zeros(1))
        self.tcpd_delta_offset = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.tcpd_delta_attn = nn.Linear(d_model, n_heads * n_levels * n_points)
        xavier_uniform_(self.tcpd_delta_offset.weight)
        constant_(self.tcpd_delta_offset.bias, 0.0)
        xavier_uniform_(self.tcpd_delta_attn.weight)
        constant_(self.tcpd_delta_attn.bias, 0.0)

        # TCPD-v2 spatial (query + target joint)
        self.tcpd_spatial_q_proj = nn.Linear(d_model, d_model)
        self.tcpd_spatial_z_proj = nn.Linear(d_model, d_model)
        self.tcpd_joint_norm = nn.LayerNorm(d_model)
        self.tcpd_spatial_delta_offset = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.tcpd_spatial_delta_attn = nn.Linear(d_model, n_heads * n_levels * n_points)
        xavier_uniform_(self.tcpd_spatial_delta_offset.weight)
        constant_(self.tcpd_spatial_delta_offset.bias, 0.0)
        xavier_uniform_(self.tcpd_spatial_delta_attn.weight)
        constant_(self.tcpd_spatial_delta_attn.bias, 0.0)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def _apply_tcpd_global(self, sampling_offsets, attention_logits, tcpd_z):
        N = tcpd_z.shape[0]
        delta_offset = self.tcpd_delta_offset(tcpd_z).view(
            N, 1, self.n_heads, self.n_levels, self.n_points, 2
        )
        sampling_offsets = sampling_offsets + self.tcpd_gate_offset * delta_offset
        delta_attn = self.tcpd_delta_attn(tcpd_z).view(
            N, 1, self.n_heads, self.n_levels * self.n_points
        )
        attention_logits = attention_logits + self.tcpd_gate_attn * delta_attn
        return sampling_offsets, attention_logits

    def _apply_tcpd_spatial(self, query, sampling_offsets, attention_logits, tcpd_z):
        N, Len_q, _ = query.shape
        z_proj = self.tcpd_spatial_z_proj(tcpd_z).unsqueeze(1)
        q_proj = self.tcpd_spatial_q_proj(query)
        joint = F.gelu(self.tcpd_joint_norm(q_proj + z_proj))
        delta_offset = self.tcpd_spatial_delta_offset(joint).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        sampling_offsets = sampling_offsets + self.tcpd_gate_offset * delta_offset
        delta_attn = self.tcpd_spatial_delta_attn(joint).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_logits = attention_logits + self.tcpd_gate_attn * delta_attn
        return sampling_offsets, attention_logits

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        tcpd_z=None,
        tcpd_spatial_mode="spatial",
        tcpd_condition_msdeform=True,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        self.sampling_offsets.bias = self.sampling_offsets.bias.to(self.sampling_offsets.weight.dtype)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_logits = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)

        if tcpd_z is not None and tcpd_condition_msdeform:
            if tcpd_spatial_mode == "spatial":
                sampling_offsets, attention_logits = self._apply_tcpd_spatial(
                    query, sampling_offsets, attention_logits, tcpd_z
                )
            else:
                sampling_offsets, attention_logits = self._apply_tcpd_global(
                    sampling_offsets, attention_logits, tcpd_z
                )

        attention_weights = F.softmax(attention_logits, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        try:
            data_type = value.dtype
            output = MSDeformAttnFunction.apply(
                value.float(), input_spatial_shapes, input_level_start_index,
                sampling_locations, attention_weights.float(), self.im2col_step,
            )
            output = output.to(data_type)
        except Exception:
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        return output

    def compute_tcpd_deltas(self, query, tcpd_z, tcpd_spatial_mode="spatial"):
        """Expose offset deltas for spatial adaptivity smoke tests."""
        if tcpd_spatial_mode == "spatial":
            z_proj = self.tcpd_spatial_z_proj(tcpd_z).unsqueeze(1)
            q_proj = self.tcpd_spatial_q_proj(query)
            joint = F.gelu(self.tcpd_joint_norm(q_proj + z_proj))
            delta_offset = self.tcpd_spatial_delta_offset(joint).view(
                query.shape[0], query.shape[1], self.n_heads, self.n_levels, self.n_points, 2
            )
            delta_attn = self.tcpd_spatial_delta_attn(joint).view(
                query.shape[0], query.shape[1], self.n_heads, self.n_levels * self.n_points
            )
        else:
            N = tcpd_z.shape[0]
            delta_offset = self.tcpd_delta_offset(tcpd_z).view(
                N, 1, self.n_heads, self.n_levels, self.n_points, 2
            ).expand(-1, query.shape[1], -1, -1, -1, -1)
            delta_attn = self.tcpd_delta_attn(tcpd_z).view(
                N, 1, self.n_heads, self.n_levels * self.n_points
            ).expand(-1, query.shape[1], -1, -1)
        return delta_offset, delta_attn
