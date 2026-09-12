"""Compile-safe boundaries for out-of-tree NPU fused operators."""

from __future__ import annotations

import torch
from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import (
    split_qkv_rmsnorm_rope as _split_qkv_rmsnorm_rope_kernel,
)

from sglang.srt.utils.custom_op import register_custom_op_from_extern


def _split_qkv_rmsnorm_rope_fake(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float | None = None,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del sin, cos, head_dim, eps, q_weight, k_weight, q_bias, k_bias, is_neox_style
    batch_size = input.shape[0]
    return (
        input.new_empty((batch_size, q_hidden_size)),
        input.new_empty((batch_size, kv_hidden_size)),
        input.new_empty((batch_size, kv_hidden_size)),
    )


def _split_qkv_rmsnorm_rope_impl(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float | None = None,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _split_qkv_rmsnorm_rope_kernel(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
        q_bias=q_bias,
        k_bias=k_bias,
        is_neox_style=is_neox_style,
    )


split_qkv_rmsnorm_rope = register_custom_op_from_extern(
    _split_qkv_rmsnorm_rope_impl,
    op_name="npu_split_qkv_rmsnorm_rope",
    mutates_args=[],
    fake_impl=_split_qkv_rmsnorm_rope_fake,
)
