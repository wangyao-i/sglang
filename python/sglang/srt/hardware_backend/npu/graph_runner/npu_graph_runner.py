# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with NPU graph and torch.compile.

NPUGraphRunner is a thin subclass of DecodeCudaGraphRunner: the
factory returns NPUCudaGraphBackend for NPU devices, so all
capture/replay mechanics live in the backend. This class adds:
  - NPU-specific patch_model monkey-patch for the decode-Full +
    torch.compile path.
  - Profile context override (NPU profiler emits to disk, not in-mem).
  - Replay override that issues an async NPUGraph.update for
    seq_lens before replay (skipped for deepseek-nsa).
  - Smaller cache_loc dtype (int32 instead of int64).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import (
    AttentionArch,
    is_deepseek_dsa,
    is_deepseek_v4,
)
from sglang.srt.compilation.torch_compile_decoration import (
    prepare_model_for_torch_compile,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.torch_compile_diagnostics import (
    get_torch_compile_diagnostic_mode,
    use_direct_graph_attention_diagnostic,
)
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.utils import (
    empty_context,
    get_bool_env_var,
    get_compiler_backend,
    is_npu,
)

is_npu = is_npu()

if is_npu:
    import torch_npu
    from torch_npu.profiler import ProfilerActivity, profile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.srt.model_loader.utils import resolve_language_model
from sglang.srt.runtime_context import get_flags


def _ensure_npu_torch_compile_layers(model_runner: ModelRunner) -> None:
    """Populate metadata required by the compile-safe attention custom op.

    Prefill graph setup normally publishes these lists. A decode-only graph
    profile skips that setup entirely, so NPU torch.compile must initialize the
    same metadata before ``DecodeCudaGraphRunner.__init__`` starts capture.
    Silently omitting the context would route capture back through eager ATB
    paged attention and recreate the failure this boundary is meant to avoid.
    """
    attention_layers = getattr(model_runner, "attention_layers", None)
    if attention_layers:
        return

    language_model = resolve_language_model(model_runner.model)
    layer_model = language_model
    while not hasattr(layer_model, "layers") and hasattr(layer_model, "model"):
        layer_model = layer_model.model
    if not hasattr(layer_model, "layers"):
        raise RuntimeError(
            "NPU torch compile requires a language model with decoder layers"
        )

    (
        model_runner.attention_layers,
        model_runner.moe_layers,
        model_runner.moe_fusions,
        model_runner.dsa_indexers,
        model_runner.mha_companion_layers,
    ) = compute_attention_and_moe_layers(layer_model)

    expected_layers = model_runner.model_config.num_hidden_layers
    if len(model_runner.attention_layers) < expected_layers:
        raise RuntimeError(
            "NPU torch compile requires attention metadata for every decoder "
            f"layer: expected {expected_layers}, found "
            f"{len(model_runner.attention_layers)}"
        )


def _diagnostic_fused_op_filter(diagnostic_mode: Optional[str]):
    if diagnostic_mode == "audio-prepared-eager":
        return lambda path, _op: path[:1] == ("audio_tower",)
    if diagnostic_mode == "language-prepared-eager":
        return lambda path, _op: path[:1] == ("language_model",)
    return None


@contextmanager
def patch_model_npu(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    if enable_compile:
        diagnostic_mode = get_torch_compile_diagnostic_mode()
        if diagnostic_mode in {"context-eager", "direct-graph-eager"}:
            logger.warning(
                "NPU torch.compile diagnostic mode %s: using raw model.forward "
                "without compile-safe fused-op dispatch",
                diagnostic_mode,
            )
            yield model.forward
            return
        module_filter = _diagnostic_fused_op_filter(diagnostic_mode)
        prepare_context = (
            prepare_model_for_torch_compile(
                model, num_tokens, tp_group, module_filter=module_filter
            )
            if module_filter is not None
            else prepare_model_for_torch_compile(model, num_tokens, tp_group)
        )
        with prepare_context:
            if diagnostic_mode == "prepared-eager":
                logger.warning(
                    "NPU torch.compile diagnostic mode prepared-eager: using "
                    "compile-safe fused-op dispatch without torch.compile"
                )
                yield model.forward
                return

            if diagnostic_mode in {
                "audio-prepared-eager",
                "language-prepared-eager",
            }:
                logger.warning(
                    "NPU torch.compile diagnostic mode %s: using raw model.forward "
                    "with a scoped compile-safe fused-op context",
                    diagnostic_mode,
                )
                yield model.forward
                return

            backend = (
                "eager"
                if diagnostic_mode == "dynamo-eager"
                else get_compiler_backend("npugraph_ex")
            )
            if diagnostic_mode:
                logger.warning(
                    "NPU torch.compile diagnostic mode %s: backend=%s",
                    diagnostic_mode,
                    backend,
                )
            yield torch.compile(
                torch.no_grad()(model.forward),
                fullgraph=True,
                dynamic=False,
                backend=backend,
            )
    else:
        yield model.forward


class NPUGraphRunner(DecodeCudaGraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with NPU graph and torch.compile."""

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        attn_backend=None,
        speculative_num_steps: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        # NPU patch_model override: monkey-patch torch_compile_decoration's
        # patch_model with the NPU-specific version.
        from sglang.srt.compilation import torch_compile_decoration

        torch_compile_decoration.patch_model = patch_model_npu
        if get_flags().capture.enable_torch_compile:
            _ensure_npu_torch_compile_layers(model_runner)
        super().__init__(
            model_runner,
            attn_backend=attn_backend,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.update_attr_name = None
        self.update_attr_type = None
        self.model_runner = model_runner
        self._init_arch_map()
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.if_use_v2 = any(
            arch
            in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "Step3p5ForCausalLM")
            for arch in (model_runner.model_config.hf_config.architectures or [])
        )

    def _init_arch_map(self):
        if self.is_dllm:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "actual_seq_lengths_kv",
                "TARGET_VERIFY": "actual_seq_kvlen",
            }
        else:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "context_lens",
                "TARGET_VERIFY": "actual_seq_kvlen",
            }
        self.attr_type: Dict[str, Union[list, torch.Tensor]] = {
            AttentionArch.MLA: [],
            AttentionArch.MHA: torch.Tensor(),
            "TARGET_VERIFY": [],
        }

    def _create_device_graph(self):
        return torch.npu.NPUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        if self.enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        with (
            skip_guard_context,
            torch.npu.graph(
                graph,
                pool=pool,
                stream=stream,
                auto_dispatch_capture=True,
            ),
        ):
            out = run_once_fn()
        return out

    def _torch_compile_forward_context(
        self, forward_batch: ForwardBatch, num_tokens: int
    ):
        if not self.enable_torch_compile:
            return empty_context()

        runner = self.model_runner
        return set_tc_piecewise_forward_context(
            forward_batch,
            runner.attention_layers,
            getattr(runner, "quant_config", None),
            getattr(runner, "moe_layers", []),
            getattr(runner, "moe_fusions", []),
            dsa_indexers=getattr(runner, "dsa_indexers", None),
            mha_companion_layers=getattr(runner, "mha_companion_layers", None),
            num_tokens=num_tokens,
            raw_num_tokens=num_tokens,
            full_graph=True,
            use_decode_graph_attention=not use_direct_graph_attention_diagnostic(),
        )

    def _get_update_attr_name(self):
        if self.if_use_v2:
            return self.attr_name["TARGET_VERIFY"]
        return self.attr_name[AttentionArch.MLA]

    def _get_update_attr_type(self):
        if self.if_use_v2:
            return self.attr_type["TARGET_VERIFY"]
        return self.attr_type[AttentionArch.MLA]

    def _update_inputs(self, seq_lens):
        if isinstance(self.update_attr_type, torch.Tensor):
            seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))

        self.graphs[self.bs].update(
            cpu_update_input=[{self.update_attr_name: seq_lens}]
        )

    def _cache_loc_dtype(self):
        return torch.int32

    def _init_profile_context_and_memory_record(self):
        output_dir = os.path.join(
            os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp"), "graph_capture_profile"
        )
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Profiling starts for graph capture for NPU. Traces will be saved to: {output_dir}"
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir, async_mode=True
            ),
            experimental_config=experimental_config,
        )
        return profile_context

    def _post_process_after_profile(self, prof_context):
        # for NPU, profile data will be saved to disk for further analysis.
        pass

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        log_graph_key = envs.SGLANG_LOG_DECODE_GRAPH_KEY.get()
        if log_graph_key:
            logger.info(
                "NPU decode graph: stage=npu_execute_begin mode=%s raw_bs=%d",
                forward_batch.forward_mode.name,
                forward_batch.batch_size,
            )
        if forward_batch.needs_forward_metadata_init():
            if log_graph_key:
                logger.info(
                    "NPU decode graph: stage=load_batch_begin mode=%s raw_bs=%d",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                )
            self.load_batch(forward_batch, pp_proxy_tensors)
            if log_graph_key:
                logger.info(
                    "NPU decode graph: stage=load_batch_return mode=%s raw_bs=%d "
                    "graph_bs=%d",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    self.bs,
                )
        else:
            if log_graph_key:
                logger.info(
                    "NPU decode graph: stage=input_copy_begin mode=%s raw_bs=%d",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                )
            # In speculative decoding, these two fields are still needed.
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
            if (
                self.model_runner.spec_algorithm.is_dflash()
                and self.model_runner.is_draft_worker
                and forward_batch.input_embeds is not None
            ):
                self.buffers.input_embeds[: self.raw_num_token].copy_(
                    forward_batch.input_embeds
                )
            if (
                envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get()
                and forward_batch.mrope_positions is not None
            ):
                self.buffers.mrope_positions[:, : self.raw_num_token].copy_(
                    forward_batch.mrope_positions
                )
            if log_graph_key:
                logger.info(
                    "NPU decode graph: stage=input_copy_return mode=%s raw_bs=%d "
                    "graph_bs=%d",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    self.bs,
                )

        graph_key = self._make_graph_key(self.bs)

        if not (
            is_deepseek_dsa(self.model_runner.model_config.hf_config)
            or is_deepseek_v4(self.model_runner.model_config.hf_config)
        ):
            if log_graph_key:
                logger.info(
                    "NPU decode graph: stage=seq_lens_host_begin mode=%s raw_bs=%d "
                    "graph_bs=%d",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    self.bs,
                )
            if forward_batch.forward_mode.is_target_verify():
                seq_lens_cpu = forward_batch.seq_lens.cpu() + self.captured_req_width
                seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
            else:
                seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                    self.bs - self.raw_bs
                )
            if log_graph_key:
                logger.info(
                    "NPU decode graph: stage=seq_lens_host_return mode=%s raw_bs=%d "
                    "graph_bs=%d",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    self.bs,
                )
                logger.info(
                    "NPU decode graph: stage=input_update_replay_begin mode=%s "
                    "raw_bs=%d graph_bs=%d key_size=%s",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    self.bs,
                    graph_key.size,
                )
            output = self.backend.replay_with_input_update(
                graph_key,
                seq_lens=seq_lens,
                attr_name=self._get_update_attr_name(),
                attr_type=self._get_update_attr_type(),
            )
            if log_graph_key:
                logger.info(
                    "NPU decode graph: stage=input_update_replay_return mode=%s "
                    "raw_bs=%d graph_bs=%d key_size=%s",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    self.bs,
                    graph_key.size,
                )
        else:
            if log_graph_key:
                logger.info(
                    "NPU decode graph: stage=backend_replay_begin mode=%s raw_bs=%d "
                    "graph_bs=%d key_size=%s",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    self.bs,
                    graph_key.size,
                )
            output = self.backend.replay(graph_key, forward_batch)
            if log_graph_key:
                logger.info(
                    "NPU decode graph: stage=backend_replay_return mode=%s raw_bs=%d "
                    "graph_bs=%d key_size=%s",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    self.bs,
                    graph_key.size,
                )

        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = (
                    output.full_logits[: self.raw_num_token]
                    if output.full_logits is not None
                    else None
                )
            else:
                full_logits = None
                next_token_logits = (
                    output.next_token_logits[: self.raw_num_token]
                    if output.next_token_logits is not None
                    else None
                )
            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})
