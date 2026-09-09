from types import SimpleNamespace

import pytest
from sglang.srt.model_executor.cuda_graph_config import (
    CudaGraphConfig,
    PhaseConfig,
    parse_cuda_graph_config_arg,
)
from sglang.srt.model_executor.runner import base_cuda_graph_runner as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _runner(*, capture_bs, compile_bs=None, max_requests=70):
    return SimpleNamespace(
        server_args=SimpleNamespace(),
        req_to_token_pool=SimpleNamespace(size=max_requests),
        _capture_bs=capture_bs,
        _compile_bs=compile_bs,
    )


def _install_runtime(monkeypatch, runner, *, max_compile_bs=2, enabled=True):
    config = CudaGraphConfig(
        decode=PhaseConfig(
            bs=runner._capture_bs,
            torch_compile_bs=runner._compile_bs,
        )
    )
    monkeypatch.setattr(
        mod,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=config,
                torch_compile_max_bs=max_compile_bs,
            ),
            overlap=SimpleNamespace(enable_two_batch_overlap=False),
        ),
    )
    monkeypatch.setattr(
        mod,
        "get_flags",
        lambda: SimpleNamespace(capture=SimpleNamespace(enable_torch_compile=enabled)),
    )
    monkeypatch.setattr(mod, "get_cuda_graph_batch_size_alignment", lambda _: 1)
    monkeypatch.setattr(mod, "get_cuda_graph_max_batch_size", lambda _, size: size)


def test_compile_batch_sizes_keep_legacy_prefix_policy(monkeypatch):
    runner = _runner(capture_bs=[1, 2, 4, 8])
    _install_runtime(monkeypatch, runner, max_compile_bs=4)

    capture_bs, compile_bs = mod.get_batch_sizes_to_capture(runner)

    assert capture_bs == [1, 2, 4, 8]
    assert compile_bs == [1, 2, 4]


def test_cuda_graph_config_accepts_explicit_compile_batch_sizes():
    raw = parse_cuda_graph_config_arg(
        '{"decode":{"backend":"full","torch_compile_bs":[1,32,70]}}'
    )

    assert raw["decode"]["torch_compile_bs"] == [1, 32, 70]


def test_compile_batch_sizes_allow_explicit_sparse_subset(monkeypatch):
    runner = _runner(
        capture_bs=[1, 2, 4, 8, 32, 64, 70],
        compile_bs=[70, 1, 32, 70],
    )
    _install_runtime(monkeypatch, runner)

    capture_bs, compile_bs = mod.get_batch_sizes_to_capture(runner)

    assert capture_bs == [1, 2, 4, 8, 32, 64, 70]
    assert compile_bs == [1, 32, 70]


@pytest.mark.parametrize("compile_bs", [[0], [True], [1.5]])
def test_compile_batch_sizes_reject_invalid_values(monkeypatch, compile_bs):
    runner = _runner(capture_bs=[1, 2, 4], compile_bs=compile_bs)
    _install_runtime(monkeypatch, runner)

    with pytest.raises(ValueError, match="positive integers"):
        mod.get_batch_sizes_to_capture(runner)


def test_compile_batch_sizes_reject_uncaptured_values(monkeypatch):
    runner = _runner(capture_bs=[1, 2, 4], compile_bs=[1, 8])
    _install_runtime(monkeypatch, runner)

    with pytest.raises(ValueError, match=r"uncaptured=\[8\]"):
        mod.get_batch_sizes_to_capture(runner)


def test_compile_batch_sizes_are_empty_when_compile_disabled(monkeypatch):
    runner = _runner(capture_bs=[1, 2, 4], compile_bs=[1, 4])
    _install_runtime(monkeypatch, runner, enabled=False)

    _, compile_bs = mod.get_batch_sizes_to_capture(runner)

    assert compile_bs == []
