from __future__ import annotations

import re

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger
from .qwen import Qwen2Model

# HF tensor suffix -> (GGUF MODEL_TENSOR attr name, GGUF name suffix)
_DSRN_TENSOR_MAP: dict[str, tuple[str, str]] = {
    "norm_fast.weight":         ("DSRN_NORM",   ".weight"),
    "gru_cell.weight_ih":       ("DSRN_GRU",    ".weight"),
    "gru_cell.bias_ih":         ("DSRN_GRU",    ".bias"),
    "linear_pred.weight":       ("DSRN_PRED",   ".weight"),
    "linear_gate.weight":       ("DSRN_GATE",   ".weight"),
    "linear_gate.bias":         ("DSRN_GATE",   ".bias"),
    "linear_memory.weight":     ("DSRN_MEM",    ".weight"),
    "linear_memory.bias":       ("DSRN_MEM",    ".bias"),
    "linear_read.weight":       ("DSRN_READ",   ".weight"),
    "surprise_lambda":          ("DSRN_LAMBDA", ""),
}

# GRUCell recurrent weights are intentionally unused (parallel-scan design) —
# retained in the checkpoint for compatibility only. Do not emit them.
_DSRN_SKIP_PATTERN = re.compile(r"model\.memory_injectors\.\d+\.gru_cell\.(weight_hh|bias_hh)$")

_DSRN_INJECTOR_PATTERN = re.compile(r"model\.memory_injectors\.(\d+)\.([a-z_.]+)$")


@ModelBase.register("HybridEchoForCausalLM")
@ModelBase.example("mrs83/Kurtis-EON1-Hybrid-0.7B-v0.1.1")
class EchoDsrnHybridModel(Qwen2Model):
    """Qwen2 backbone + DSRN memory injectors (additive residual blocks).

    The checkpoint wraps the Qwen2 backbone under ``model.backbone.*`` and the
    injectors under ``model.memory_injectors.{j}.*``.  Injector ``j`` runs after
    transformer layer ``(j+1) * stride - 1``, so its tensors are stored under
    ``blk.{layer}.dsrn_*`` with the same layer index as the backbone block.
    """

    model_arch = gguf.MODEL_ARCH.ECHO_DSRN_HYBRID

    def set_vocab(self) -> None:
        # Kurtis-EON1 ships a BPE tokenizer (tokenizer.json) with no
        # sentencepiece model file — go straight to the GPT2-style path.
        self._set_vocab_gpt2()

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        state_dim = self.find_hparam(["dsrn_state_dim"], optional=True)
        if state_dim is None:
            state_dim = 512
        stride = self.find_hparam(["dsrn_injection_stride"], optional=True)
        if stride is None:
            stride = 4
        self.gguf_writer.add_dsrn_state_dim(state_dim)
        self.gguf_writer.add_dsrn_injection_stride(stride)
        logger.info(f"gguf: dsrn state dim = {state_dim}, injection stride = {stride}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Skip unused GRUCell recurrent weights (parallel-scan design).
        if _DSRN_SKIP_PATTERN.match(name):
            return []

        # DSRN injector tensors: model.memory_injectors.{j}.{sub} -> blk.{layer}.dsrn_{...}
        m = _DSRN_INJECTOR_PATTERN.match(name)
        if m is not None:
            injector_idx = int(m.group(1))
            sub = m.group(2)
            entry = _DSRN_TENSOR_MAP.get(sub)
            if entry is None:
                raise ValueError(f"Unexpected DSRN injector tensor {name!r}")
            tensor_key, suffix = entry
            stride = self.find_hparam(["dsrn_injection_stride"], optional=True) or 4
            layer = (injector_idx + 1) * stride - 1
            new_name = self.format_tensor_name(
                getattr(gguf.MODEL_TENSOR, tensor_key), layer, suffix=suffix
            )
            logger.info(f"gguf: DSRN injector {injector_idx} -> {new_name} (layer {layer})")
            return [(new_name, data_torch)]

        # Qwen2 backbone lives under model.backbone.* in this checkpoint.
        if name.startswith("model.backbone."):
            name = name.removeprefix("model.backbone.")

        return super().modify_tensors(data_torch, name, bid)
