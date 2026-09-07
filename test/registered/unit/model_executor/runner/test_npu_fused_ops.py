from __future__ import annotations

import pytest
import torch

pytest.importorskip("sgl_kernel_npu")

from sglang.srt.hardware_backend.npu.fused_ops import (
    _split_qkv_rmsnorm_rope_fake,
    split_qkv_rmsnorm_rope,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx


def test_split_qkv_rmsnorm_rope_fake_preserves_output_metadata():
    with FakeTensorMode():
        qkv = torch.empty((1, 4096), dtype=torch.bfloat16)
        sin = torch.empty((1, 128), dtype=torch.bfloat16)
        cos = torch.empty((1, 128), dtype=torch.bfloat16)
        weight = torch.empty((128,), dtype=torch.bfloat16)

        q, k, v = _split_qkv_rmsnorm_rope_fake(
            qkv,
            sin,
            cos,
            q_hidden_size=3072,
            kv_hidden_size=512,
            head_dim=128,
            eps=1e-6,
            q_weight=weight,
            k_weight=weight,
        )

    assert q.shape == (1, 3072)
    assert k.shape == (1, 512)
    assert v.shape == (1, 512)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16


def test_split_qkv_rmsnorm_rope_is_registered_as_custom_op():
    assert hasattr(torch.ops.sglang, "npu_split_qkv_rmsnorm_rope")


def test_split_qkv_rmsnorm_rope_is_opaque_to_symbolic_trace():
    def call_op(qkv, sin, cos, weight):
        return split_qkv_rmsnorm_rope(
            qkv,
            sin,
            cos,
            q_hidden_size=3072,
            kv_hidden_size=512,
            head_dim=128,
            eps=1e-6,
            q_weight=weight,
            k_weight=weight,
        )

    graph = make_fx(call_op, tracing_mode="fake")(
        torch.empty((1, 4096), dtype=torch.bfloat16),
        torch.empty((1, 128), dtype=torch.bfloat16),
        torch.empty((1, 128), dtype=torch.bfloat16),
        torch.empty((128,), dtype=torch.bfloat16),
    )

    call_targets = [
        node.target for node in graph.graph.nodes if node.op == "call_function"
    ]
    assert torch.ops.sglang.npu_split_qkv_rmsnorm_rope.default in call_targets
