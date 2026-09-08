"""Opt-in NPU Torch-Compile diagnostics.

These selectors are intentionally process-scoped and must remain inert in
normal serving.  They allow correctness localization on inaccessible NPU
hardware without changing the production dispatch policy.
"""

from __future__ import annotations

import os
from typing import Optional

TORCH_COMPILE_DIAGNOSTIC_ENV = "SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC"
TORCH_COMPILE_DIAGNOSTIC_MODES = frozenset(
    {
        "prepared-eager",
        "dynamo-eager",
        "context-eager",
        "audio-prepared-eager",
        "language-prepared-eager",
        # Keep the TC context and NPU graph backend, but bypass the
        # unified-attention custom-op wrapper. This isolates the wrapper's
        # metadata/output handling from AscendAttentionBackend.forward_decode_graph.
        "direct-graph-eager",
    }
)


def get_torch_compile_diagnostic_mode() -> Optional[str]:
    """Return a validated opt-in diagnostic mode, if one was requested."""
    diagnostic_mode = os.environ.get(TORCH_COMPILE_DIAGNOSTIC_ENV)
    if diagnostic_mode and diagnostic_mode not in TORCH_COMPILE_DIAGNOSTIC_MODES:
        raise ValueError(
            f"Unsupported {TORCH_COMPILE_DIAGNOSTIC_ENV}={diagnostic_mode!r}; "
            f"expected one of {sorted(TORCH_COMPILE_DIAGNOSTIC_MODES)}"
        )
    return diagnostic_mode


def use_direct_graph_attention_diagnostic() -> bool:
    """Whether a diagnostic run must bypass the TC attention custom op."""
    return get_torch_compile_diagnostic_mode() == "direct-graph-eager"
