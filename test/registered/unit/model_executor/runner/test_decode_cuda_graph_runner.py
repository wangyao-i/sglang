"""Unit tests for ``DecodeCudaGraphRunner`` capture-phase profiling — CPU-only.

Two capture-trace modes plus their precedence:

  * **Original single-trace** (``SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE``):
    ``_init_profile_context_and_memory_record`` builds an *unscheduled* profiler
    (``record_shapes`` only, no schedule / no ``on_trace_ready``); the combined
    trace is exported in ``_post_process_after_profile`` via
    ``export_cuda_graph_capture_trace``.
  * **Per-batch-size traces** (``SGLANG_GRAPH_BATCH_CAPTURE``): a *scheduled*
    profiler (``wait=2, warmup=0, active=1, repeat=0``) with the trace-export
    knobs (record_shapes / with_stack / with_flops / profile_memory) and an
    ``on_trace_ready`` hook that writes one trace per batch size to
    ``<SGLANG_TORCH_PROFILER_DIR>/graph_capture_profile/`` named
    ``{runner_name}_bs_{bs}_rank{rank}.json.gz``.
  * **Precedence**: when both env vars are set, the original single-trace path
    wins (no per-bs schedule / dir / bookkeeping).

The profiler / CUDA-memory APIs are mocked; the directory + naming + schedule
logic is pure-Python and runs on CPU. The method is invoked unbound against a
lightweight stand-in (with the real precedence helper bound) so no model or
server is constructed.
"""

import os
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.utils import profile_utils as putils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CAPTURE_TRACE = "SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE"
_BATCH_CAPTURE = "SGLANG_GRAPH_BATCH_CAPTURE"


def test_compile_safe_model_context_restores_fused_ops_and_ca_comm():
    from sglang.srt.compilation import torch_compile_decoration

    model = object()
    original_ca_comm = object()
    tp_group = SimpleNamespace(ca_comm=original_ca_comm)
    with mock.patch.object(torch_compile_decoration, "_to_torch") as toggle:
        with torch_compile_decoration.prepare_model_for_torch_compile(
            model, num_tokens=4, tp_group=tp_group
        ):
            tp_group.ca_comm = object()
            toggle.assert_called_once_with(model, reverse=False, num_tokens=4)

    assert toggle.call_args_list == [
        mock.call(model, reverse=False, num_tokens=4),
        mock.call(model, reverse=True, num_tokens=4),
    ]
    assert tp_group.ca_comm is original_ca_comm


def test_npu_patch_model_uses_compile_safe_model_context():
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner

    events = []

    @contextmanager
    def fake_prepare(model, num_tokens, tp_group):
        events.append(("enter", model, num_tokens, tp_group))
        try:
            yield
        finally:
            events.append(("exit", model, num_tokens, tp_group))

    model = SimpleNamespace(forward=lambda *args, **kwargs: None)
    tp_group = SimpleNamespace(ca_comm=object())
    compiled = object()
    with (
        mock.patch.object(
            npu_graph_runner, "prepare_model_for_torch_compile", fake_prepare
        ),
        mock.patch.object(
            npu_graph_runner, "get_compiler_backend", return_value="npugraph_ex"
        ),
        mock.patch.object(
            npu_graph_runner.torch, "compile", return_value=compiled
        ) as compile_mock,
    ):
        with npu_graph_runner.patch_model_npu(
            model, True, num_tokens=8, tp_group=tp_group
        ) as forward:
            assert forward is compiled
            assert [event[0] for event in events] == ["enter"]

    assert [event[0] for event in events] == ["enter", "exit"]
    assert compile_mock.call_args.kwargs == {
        "fullgraph": False,
        "dynamic": False,
        "backend": "npugraph_ex",
    }


def test_npu_patch_model_prepared_eager_diagnostic_skips_torch_compile(monkeypatch):
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner

    model = SimpleNamespace(forward=lambda *args, **kwargs: None)
    tp_group = SimpleNamespace(ca_comm=object())
    monkeypatch.setenv(
        "SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC", "prepared-eager"
    )

    with mock.patch.object(
        npu_graph_runner,
        "prepare_model_for_torch_compile",
        lambda *args: nullcontext(),
    ), mock.patch.object(npu_graph_runner.torch, "compile") as compile_mock:
        with npu_graph_runner.patch_model_npu(
            model, True, num_tokens=8, tp_group=tp_group
        ) as forward:
            assert forward is model.forward

    compile_mock.assert_not_called()


def test_npu_patch_model_context_eager_diagnostic_skips_compile_safe_dispatch(
    monkeypatch,
):
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner

    model = SimpleNamespace(forward=lambda *args, **kwargs: None)
    tp_group = SimpleNamespace(ca_comm=object())
    monkeypatch.setenv("SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC", "context-eager")

    with mock.patch.object(
        npu_graph_runner, "prepare_model_for_torch_compile"
    ) as prepare, mock.patch.object(npu_graph_runner.torch, "compile") as compile_mock:
        with npu_graph_runner.patch_model_npu(
            model, True, num_tokens=8, tp_group=tp_group
        ) as forward:
            assert forward is model.forward

    prepare.assert_not_called()
    compile_mock.assert_not_called()


@pytest.mark.parametrize(
    ("diagnostic_mode", "expected_root"),
    [
        ("audio-prepared-eager", "audio_tower"),
        ("language-prepared-eager", "language_model"),
    ],
)
def test_npu_patch_model_scoped_prepared_eager_diagnostic(
    monkeypatch, diagnostic_mode, expected_root
):
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner

    model = SimpleNamespace(forward=lambda *args, **kwargs: None)
    tp_group = SimpleNamespace(ca_comm=object())
    monkeypatch.setenv("SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC", diagnostic_mode)

    with mock.patch.object(
        npu_graph_runner,
        "prepare_model_for_torch_compile",
        return_value=nullcontext(),
    ) as prepare, mock.patch.object(npu_graph_runner.torch, "compile") as compile_mock:
        with npu_graph_runner.patch_model_npu(
            model, True, num_tokens=8, tp_group=tp_group
        ) as forward:
            assert forward is model.forward

    module_filter = prepare.call_args.kwargs["module_filter"]
    assert module_filter((expected_root, "block"), object())
    assert not module_filter(("other", "block"), object())
    compile_mock.assert_not_called()


def test_npu_patch_model_dynamo_eager_diagnostic_uses_eager_backend(monkeypatch):
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner

    model = SimpleNamespace(forward=lambda *args, **kwargs: None)
    tp_group = SimpleNamespace(ca_comm=object())
    compiled = object()
    monkeypatch.setenv("SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC", "dynamo-eager")

    with mock.patch.object(
        npu_graph_runner,
        "prepare_model_for_torch_compile",
        lambda *args: nullcontext(),
    ), mock.patch.object(
        npu_graph_runner.torch, "compile", return_value=compiled
    ) as compile_mock:
        with npu_graph_runner.patch_model_npu(
            model, True, num_tokens=8, tp_group=tp_group
        ) as forward:
            assert forward is compiled

    assert compile_mock.call_args.kwargs == {
        "fullgraph": False,
        "dynamic": False,
        "backend": "eager",
    }


def test_npu_patch_model_rejects_unknown_compile_diagnostic(monkeypatch):
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner

    monkeypatch.setenv("SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC", "invalid")
    with pytest.raises(
        ValueError, match="Unsupported SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC"
    ):
        with npu_graph_runner.patch_model_npu(
            SimpleNamespace(forward=lambda *args, **kwargs: None),
            True,
            num_tokens=8,
            tp_group=SimpleNamespace(ca_comm=object()),
        ):
            pass


def test_npu_compile_layer_metadata_is_initialized_without_prefill_graph():
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner

    attention_layers = [object(), object()]
    decoder_layers = [
        SimpleNamespace(self_attn=SimpleNamespace(attn=attention_layer))
        for attention_layer in attention_layers
    ]
    model_runner = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(layers=decoder_layers)
            )
        ),
        model_config=SimpleNamespace(num_hidden_layers=2),
    )

    npu_graph_runner._ensure_npu_torch_compile_layers(model_runner)

    assert model_runner.attention_layers == attention_layers
    assert model_runner.moe_layers == [None, None]
    assert model_runner.moe_fusions == [None, None]
    assert model_runner.dsa_indexers == [None, None]
    assert model_runner.mha_companion_layers == [None, None]


def test_npu_compile_layer_metadata_preserves_existing_setup():
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner

    attention_layers = [object()]
    model_runner = SimpleNamespace(attention_layers=attention_layers)

    npu_graph_runner._ensure_npu_torch_compile_layers(model_runner)

    assert model_runner.attention_layers is attention_layers


def test_npu_compile_layer_metadata_rejects_incomplete_attention_map():
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner

    model_runner = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(
                    layers=[SimpleNamespace(mlp=SimpleNamespace())]
                )
            )
        ),
        model_config=SimpleNamespace(num_hidden_layers=1),
    )

    with pytest.raises(
        RuntimeError,
        match="expected 1, found 0",
    ):
        npu_graph_runner._ensure_npu_torch_compile_layers(model_runner)


def test_npu_compile_context_requests_graph_safe_decode_attention():
    from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
        NPUGraphRunner,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    forward_batch = object()
    attention_layers = [object()]
    fake_self = SimpleNamespace(
        enable_torch_compile=True,
        model_runner=SimpleNamespace(
            attention_layers=attention_layers,
            quant_config=None,
            moe_layers=[],
            moe_fusions=[],
            dsa_indexers=None,
            mha_companion_layers=None,
        ),
    )

    with NPUGraphRunner._torch_compile_forward_context(
        fake_self, forward_batch, num_tokens=4
    ):
        context = get_tc_piecewise_forward_context()
        assert context.forward_batch is forward_batch
        assert context.attention_layers is attention_layers
        assert context.num_tokens == 4
        assert context.raw_num_tokens == 4
        assert context.full_graph is True
        assert context.use_decode_graph_attention is True
        assert context.use_explicit_decode_attention_state is False

    assert get_tc_piecewise_forward_context() is None


def test_direct_graph_attention_diagnostic_bypasses_tc_attention_custom_op():
    from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
        NPUGraphRunner,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    forward_batch = object()
    fake_self = SimpleNamespace(
        enable_torch_compile=True,
        model_runner=SimpleNamespace(
            attention_layers=[object()],
            quant_config=None,
            moe_layers=[],
            moe_fusions=[],
            dsa_indexers=None,
            mha_companion_layers=None,
        ),
    )

    with mock.patch.dict(
        os.environ,
        {"SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC": "direct-graph-eager"},
    ):
        with NPUGraphRunner._torch_compile_forward_context(
            fake_self, forward_batch, num_tokens=4
        ):
            context = get_tc_piecewise_forward_context()
            assert context is not None
            assert context.full_graph is True
            assert context.use_decode_graph_attention is False

    assert get_tc_piecewise_forward_context() is None


def test_npu_torch_compile_diagnostic_mode_is_validated():
    from sglang.srt.hardware_backend.npu.graph_runner.torch_compile_diagnostics import (
        get_torch_compile_diagnostic_mode,
        use_direct_graph_attention_diagnostic,
        use_explicit_state_attention_diagnostic,
    )

    with mock.patch.dict(os.environ, {}, clear=True):
        assert get_torch_compile_diagnostic_mode() is None
        assert use_direct_graph_attention_diagnostic() is False
        assert use_explicit_state_attention_diagnostic() is False

    with mock.patch.dict(
        os.environ,
        {"SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC": "direct-graph-eager"},
        clear=True,
    ):
        assert get_torch_compile_diagnostic_mode() == "direct-graph-eager"
        assert use_direct_graph_attention_diagnostic() is True
        assert use_explicit_state_attention_diagnostic() is False

    with mock.patch.dict(
        os.environ,
        {"SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC": "explicit-state-attention"},
        clear=True,
    ):
        assert get_torch_compile_diagnostic_mode() == "explicit-state-attention"
        assert use_direct_graph_attention_diagnostic() is False
        assert use_explicit_state_attention_diagnostic() is True

    with mock.patch.dict(
        os.environ,
        {"SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC": "not-a-mode"},
        clear=True,
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            get_torch_compile_diagnostic_mode()


def test_explicit_state_attention_diagnostic_reaches_compile_context():
    from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
        NPUGraphRunner,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    fake_self = SimpleNamespace(
        enable_torch_compile=True,
        model_runner=SimpleNamespace(
            attention_layers=[object()],
            quant_config=None,
            moe_layers=[],
            moe_fusions=[],
            dsa_indexers=None,
            mha_companion_layers=None,
        ),
    )
    with mock.patch.dict(
        os.environ,
        {"SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC": "explicit-state-attention"},
        clear=True,
    ):
        with NPUGraphRunner._torch_compile_forward_context(
            fake_self, object(), num_tokens=4
        ):
            context = get_tc_piecewise_forward_context()
            assert context is not None
            assert context.use_decode_graph_attention is True
            assert context.use_explicit_decode_attention_state is True


def test_compile_safe_attention_calls_decode_graph_implementation():
    from sglang.srt.layers import radix_attention

    mode = SimpleNamespace(is_decode=lambda: True)
    forward_batch = SimpleNamespace(
        forward_mode=mode,
        num_token_non_padded_cpu=2,
        out_cache_loc=torch.arange(2),
        positions=torch.arange(2),
    )
    layer = SimpleNamespace()
    context = SimpleNamespace(
        forward_batch=forward_batch,
        attention_layers=[layer],
        mha_companion_layers=None,
        use_decode_graph_attention=True,
        num_tokens=2,
        raw_num_tokens=2,
    )
    output = torch.empty((2, 4))
    backend = SimpleNamespace()

    def forward_decode_graph(query, key, value, *args, **kwargs):
        output.fill_(3)
        return output

    backend.forward_decode_graph = mock.Mock(side_effect=forward_decode_graph)
    backend.forward = mock.Mock(side_effect=AssertionError("eager attention selected"))

    with (
        mock.patch.object(
            radix_attention, "get_tc_piecewise_forward_context", return_value=context
        ),
        mock.patch.object(radix_attention, "get_attn_backend", return_value=backend),
    ):
        radix_attention._unified_attention_with_output_impl(
            torch.zeros((2, 4)),
            torch.zeros((2, 4)),
            torch.zeros((2, 4)),
            output,
            True,
            0,
            False,
            False,
        )

    backend.forward_decode_graph.assert_called_once()
    backend.forward.assert_not_called()
    assert torch.equal(output, torch.full_like(output, 3))


def test_npu_prefill_capture_only_diagnostic_gate(monkeypatch):
    from sglang.srt.model_executor import model_runner

    monkeypatch.setattr(model_runner, "_is_npu", True)
    monkeypatch.delenv("SGLANG_NPU_PREFILL_GRAPH_CAPTURE_ONLY", raising=False)
    assert not model_runner._skip_npu_prefill_graph_replay_for_diagnostics()

    monkeypatch.setenv("SGLANG_NPU_PREFILL_GRAPH_CAPTURE_ONLY", "true")
    assert model_runner._skip_npu_prefill_graph_replay_for_diagnostics()


def test_decode_graph_diagnostics_cover_dispatch_and_replay_boundaries():
    root = Path(__file__).resolve().parents[5]
    model_runner_source = (
        root / "python/sglang/srt/model_executor/model_runner.py"
    ).read_text(encoding="utf-8")
    decode_runner_source = (
        root / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
    ).read_text(encoding="utf-8")
    npu_runner_source = (
        root / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py"
    ).read_text(encoding="utf-8")
    npu_backend_source = (
        root
        / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py"
    ).read_text(encoding="utf-8")

    for stage in (
        "eligibility_begin",
        "eligibility_return",
        "execute_begin",
        "execute_return",
        "forward_raw_return",
        "model_forward_return",
    ):
        assert model_runner_source.count(f"stage={stage}") == 1
    for stage in (
        "runner_enter",
        "replay_session_enter",
        "load_batch_return",
        "backend_replay_begin",
        "backend_replay_return",
        "replay_session_return",
    ):
        assert decode_runner_source.count(f"stage={stage}") == 1
    for stage in (
        "npu_execute_begin",
        "load_batch_begin",
        "load_batch_return",
        "input_copy_begin",
        "input_copy_return",
        "seq_lens_host_begin",
        "seq_lens_host_return",
        "input_update_replay_begin",
        "input_update_replay_return",
        "backend_replay_begin",
        "backend_replay_return",
    ):
        assert npu_runner_source.count(f"stage={stage}") == 1
    for stage in (
        "backend_enter",
        "cpu_update_input_ready",
        "update_thread_start_begin",
        "update_thread_start_return",
        "update_thread_enter",
        "update_device_set",
        "graph_update_begin",
        "graph_update_return",
        "graph_replay_begin",
        "graph_replay_return",
        "update_thread_join_begin",
        "update_thread_join_return",
    ):
        assert npu_backend_source.count(f"stage={stage}") == 1


def _make_fake_self(capture_bs):
    """Stand-in ``self`` with the real precedence helper bound so the env-var
    gating in ``_init_profile_context_and_memory_record`` applies."""
    fake_self = SimpleNamespace(capture_bs=list(capture_bs))
    fake_self._graph_batch_capture_active = (
        DecodeCudaGraphRunner._graph_batch_capture_active.__get__(fake_self)
    )
    return fake_self


class TestInitProfileBatchMode(CustomTestCase):
    """SGLANG_GRAPH_BATCH_CAPTURE -> scheduled per-bs profiler."""

    def _invoke(self, *, capture_bs, rank=0, profiler_dir=None):
        fake_self = _make_fake_self(capture_bs)
        env = {_BATCH_CAPTURE: "1"}
        if profiler_dir is not None:
            env["SGLANG_TORCH_PROFILER_DIR"] = profiler_dir
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                mod, "get_parallel", return_value=SimpleNamespace(tp_rank=rank)
            ),
            mock.patch.object(mod, "profile") as mock_profile,
            mock.patch("torch.profiler.schedule") as mock_schedule,
            mock.patch(
                "torch.cuda.memory._record_memory_history"
            ) as mock_record_history,
        ):
            os.environ.pop(_CAPTURE_TRACE, None)  # original flag off
            if profiler_dir is None:
                os.environ.pop("SGLANG_TORCH_PROFILER_DIR", None)
            ctx = DecodeCudaGraphRunner._init_profile_context_and_memory_record(
                fake_self
            )
        self.assertIs(ctx, mock_profile.return_value)
        return fake_self, mock_profile, mock_schedule, mock_record_history

    def test_creates_graph_capture_profile_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._invoke(capture_bs=[1, 2, 4], profiler_dir=tmp)
            self.assertTrue(os.path.isdir(os.path.join(tmp, "graph_capture_profile")))

    def test_primes_reversed_bs_list_and_zero_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, *_ = self._invoke(capture_bs=[1, 2, 4, 8], profiler_dir=tmp)
            # Capture iterates large -> small, so the bs list is reversed.
            self.assertEqual(fake_self._profile_bs_list, [8, 4, 2, 1])
            self.assertEqual(fake_self._profile_bs_idx, 0)

    def test_profiler_built_with_trace_export_knobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, mock_profile, mock_schedule, mock_record_history = self._invoke(
                capture_bs=[1, 2], profiler_dir=tmp
            )
            self.assertEqual(mock_profile.call_count, 1)
            kwargs = mock_profile.call_args.kwargs
            self.assertTrue(kwargs["record_shapes"])
            self.assertTrue(kwargs["with_stack"])
            self.assertTrue(kwargs["with_flops"])
            self.assertTrue(kwargs["profile_memory"])
            self.assertTrue(callable(kwargs["on_trace_ready"]))
            # Schedule skips the two dummy/warmup runs and records the capture.
            mock_schedule.assert_called_once_with(wait=2, warmup=0, active=1, repeat=0)
            self.assertIs(kwargs["schedule"], mock_schedule.return_value)
            # Memory history recording is armed alongside the profiler.
            mock_record_history.assert_called_once()

    def test_default_dir_used_when_profiler_dir_env_unset(self):
        # No SGLANG_TORCH_PROFILER_DIR -> falls back to the envs default base dir.
        # Patch makedirs so the test never writes to the cwd.
        fake_self = _make_fake_self([1])
        with (
            mock.patch.dict(os.environ, {_BATCH_CAPTURE: "1"}, clear=False),
            mock.patch.object(
                mod, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
            ),
            mock.patch.object(mod, "profile"),
            mock.patch("torch.profiler.schedule"),
            mock.patch("torch.cuda.memory._record_memory_history"),
            mock.patch.object(mod.os, "makedirs") as mock_makedirs,
        ):
            os.environ.pop("SGLANG_TORCH_PROFILER_DIR", None)
            os.environ.pop(_CAPTURE_TRACE, None)
            DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)

        mock_makedirs.assert_called_once()
        self.assertEqual(
            mock_makedirs.call_args.args[0],
            os.path.join("/tmp", "graph_capture_profile"),
        )


class TestInitProfileOriginalMode(CustomTestCase):
    """No flag, original flag only, or both (precedence) -> unscheduled pass with
    no per-bs schedule / directory / bookkeeping."""

    def _invoke_original(self, *, env):
        fake_self = _make_fake_self([1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            environ = dict(env)
            environ["SGLANG_TORCH_PROFILER_DIR"] = tmp
            with (
                mock.patch.dict(os.environ, environ, clear=False),
                mock.patch.object(
                    mod, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
                ),
                mock.patch.object(mod, "profile") as mock_profile,
                mock.patch("torch.profiler.schedule") as mock_schedule,
                mock.patch("torch.cuda.memory._record_memory_history"),
            ):
                for k in (_CAPTURE_TRACE, _BATCH_CAPTURE):
                    if k not in environ:
                        os.environ.pop(k, None)
                DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)
            kwargs = mock_profile.call_args.kwargs
            # Unscheduled pass: record_shapes only, no schedule / on_trace_ready.
            self.assertTrue(kwargs["record_shapes"])
            self.assertIsNone(kwargs.get("schedule"))
            self.assertIsNone(kwargs.get("on_trace_ready"))
            mock_schedule.assert_not_called()
            self.assertFalse(os.path.isdir(os.path.join(tmp, "graph_capture_profile")))
            self.assertFalse(hasattr(fake_self, "_profile_bs_list"))

    def test_no_flags(self):
        self._invoke_original(env={})

    def test_original_flag_only(self):
        self._invoke_original(env={_CAPTURE_TRACE: "1"})

    def test_both_flags_original_takes_precedence(self):
        self._invoke_original(env={_CAPTURE_TRACE: "1", _BATCH_CAPTURE: "1"})


class TestOnTraceReadyNaming(CustomTestCase):
    def _build_on_trace_ready(self, *, capture_bs, rank, tmp):
        fake_self = _make_fake_self(capture_bs)
        with (
            mock.patch.dict(
                os.environ,
                {"SGLANG_TORCH_PROFILER_DIR": tmp, _BATCH_CAPTURE: "1"},
                clear=False,
            ),
            mock.patch.object(
                mod, "get_parallel", return_value=SimpleNamespace(tp_rank=rank)
            ),
            mock.patch.object(mod, "profile") as mock_profile,
            mock.patch("torch.profiler.schedule"),
            mock.patch("torch.cuda.memory._record_memory_history"),
        ):
            os.environ.pop(_CAPTURE_TRACE, None)
            DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)
        on_trace_ready = mock_profile.call_args.kwargs["on_trace_ready"]
        return fake_self, on_trace_ready

    def test_exports_one_named_trace_per_bs_and_advances_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_bs = [1, 2, 4]  # reversed -> [4, 2, 1]
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=capture_bs, rank=0, tmp=tmp
            )
            trace_dir = os.path.join(tmp, "graph_capture_profile")
            runner = type(fake_self).__name__

            exported = []
            for expected_bs in [4, 2, 1]:
                prof = mock.Mock()
                prof.export_chrome_trace.side_effect = lambda p: exported.append(p)
                on_trace_ready(prof)
                prof.export_chrome_trace.assert_called_once_with(
                    os.path.join(trace_dir, f"{runner}_bs_{expected_bs}_rank0.json.gz")
                )

            self.assertEqual(
                exported,
                [
                    os.path.join(trace_dir, f"{runner}_bs_4_rank0.json.gz"),
                    os.path.join(trace_dir, f"{runner}_bs_2_rank0.json.gz"),
                    os.path.join(trace_dir, f"{runner}_bs_1_rank0.json.gz"),
                ],
            )
            # Index advanced once per flush.
            self.assertEqual(fake_self._profile_bs_idx, 3)

    def test_rank_in_trace_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=[8], rank=3, tmp=tmp
            )
            runner = type(fake_self).__name__
            prof = mock.Mock()
            on_trace_ready(prof)
            prof.export_chrome_trace.assert_called_once_with(
                os.path.join(
                    tmp, "graph_capture_profile", f"{runner}_bs_8_rank3.json.gz"
                )
            )


class TestOriginalTraceExport(CustomTestCase):
    """export_cuda_graph_capture_trace (original single combined trace per rank),
    gated by SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE, and the shared dir helper.
    Both trace modes land under graph_capture_profile/."""

    def test_writes_named_trace_when_flag_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"SGLANG_TORCH_PROFILER_DIR": tmp, _CAPTURE_TRACE: "1"},
                clear=False,
            ):
                prof = mock.Mock()
                putils.export_cuda_graph_capture_trace(
                    prof, runner_name="DecodeCudaGraphRunner", tp_rank=2
                )
                expected = os.path.join(
                    tmp,
                    "graph_capture_profile",
                    "cuda_graph_capture-DecodeCudaGraphRunner-TP-2.json.gz",
                )
                prof.export_chrome_trace.assert_called_once_with(expected)
                self.assertTrue(
                    os.path.isdir(os.path.join(tmp, "graph_capture_profile"))
                )

    def test_noop_when_flag_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ, {"SGLANG_TORCH_PROFILER_DIR": tmp}, clear=False
            ):
                os.environ.pop(_CAPTURE_TRACE, None)
                prof = mock.Mock()
                putils.export_cuda_graph_capture_trace(
                    prof, runner_name="DecodeCudaGraphRunner", tp_rank=0
                )
                prof.export_chrome_trace.assert_not_called()
                self.assertFalse(
                    os.path.isdir(os.path.join(tmp, "graph_capture_profile"))
                )

    def test_dir_helper_uses_profiler_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ, {"SGLANG_TORCH_PROFILER_DIR": tmp}, clear=False
            ):
                self.assertEqual(
                    putils.graph_capture_profile_dir(),
                    os.path.join(tmp, "graph_capture_profile"),
                )


if __name__ == "__main__":
    unittest.main()
