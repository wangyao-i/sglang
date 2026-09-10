from __future__ import annotations

import contextlib

import pytest

from sglang.srt.model_executor.model_runner import ModelRunner


def test_external_graph_execution_context_defaults_to_noop() -> None:
    runner = ModelRunner.__new__(ModelRunner)

    with runner._external_graph_execution_context("decode"):
        pass


def test_external_graph_execution_context_forwards_phase() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    events: list[str] = []

    @contextlib.contextmanager
    def context_for(phase: str):
        events.append(f"{phase}:enter")
        try:
            yield
        finally:
            events.append(f"{phase}:exit")

    runner._external_graph_execution_context_factory = context_for

    with runner._external_graph_execution_context("prefill"):
        events.append("body")

    assert events == ["prefill:enter", "body", "prefill:exit"]


def test_external_graph_execution_context_rejects_none() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner._external_graph_execution_context_factory = lambda _phase: None

    with pytest.raises(RuntimeError, match="phase=decode"):
        runner._external_graph_execution_context("decode")


def test_external_model_execution_context_defaults_to_noop() -> None:
    runner = ModelRunner.__new__(ModelRunner)

    with runner._external_model_execution_context("decode"):
        pass


def test_external_model_execution_context_forwards_phase() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    events: list[str] = []

    @contextlib.contextmanager
    def context_for(phase: str):
        events.append(f"{phase}:enter")
        try:
            yield
        finally:
            events.append(f"{phase}:exit")

    runner._external_model_execution_context_factory = context_for

    with runner._external_model_execution_context("prefill"):
        events.append("body")

    assert events == ["prefill:enter", "body", "prefill:exit"]


def test_external_model_execution_context_rejects_none() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner._external_model_execution_context_factory = lambda _phase: None

    with pytest.raises(RuntimeError, match="phase=decode"):
        runner._external_model_execution_context("decode")
