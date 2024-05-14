# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional

import torch
from torchao.quantization.quant_api import (
    apply_weight_only_int8_quant,
    Int4WeightOnlyGPTQQuantizer,
    Int4WeightOnlyQuantizer,
    Quantizer,
)
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_3

__all__ = [
    "Int4WeightOnlyQuantizer",
    "Int4WeightOnlyGPTQQuantizer",
    "Int8WeightOnlyQuantizer",
    "get_quantizer_mode",
]


class Int8WeightOnlyQuantizer(Quantizer):
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        apply_weight_only_int8_quant(model)
        return model


_quantizer_to_mode = {
    Int4WeightOnlyQuantizer: "4w",
    Int8WeightOnlyQuantizer: "8w",
    Int4WeightOnlyGPTQQuantizer: "4w-gptq",
}
_quantizer_mode_to_disable_fake_quant = {}
_quantizer_mode_to_enable_fake_quant = {}


if TORCH_VERSION_AFTER_2_3:
    from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer
    from torchao.quantization.prototype.qat import (
        disable_8da4w_fake_quant,
        enable_8da4w_fake_quant,
        Int8DynActInt4WeightQATQuantizer,
    )

    __all__.append("Int8DynActInt4WeightQuantizer")
    __all__.append("Int8DynActInt4WeightQATQuantizer")
    _quantizer_to_mode[Int8DynActInt4WeightQuantizer] = "8da4w"
    _quantizer_to_mode[Int8DynActInt4WeightQATQuantizer] = "8da4w-qat"
    _quantizer_mode_to_disable_fake_quant["8da4w-qat"] = disable_8da4w_fake_quant
    _quantizer_mode_to_enable_fake_quant["8da4w-qat"] = enable_8da4w_fake_quant


def get_quantizer_mode(quantizer: Optional[Callable]) -> Optional[str]:
    """Given a quantizer object, returns a string that specifies the type of quantization e.g.
    4w, which means int4 weight only quantization.
    If the quantizer is not recognized as a known quantizer, we'll return None
    """
    return _quantizer_to_mode.get(type(quantizer), None)


def _get_disable_fake_quant(quantizer_mode: str) -> Callable:
    """Given a quantizer mode, return the corresponding function for disabling fake
    quantize in a model prepared by the quantizer.
    If the quantizer is not recognized as a known QAT quantizer, we'll return None.
    """
    return _quantizer_mode_to_disable_fake_quant.get(quantizer_mode, None)

def _get_enable_fake_quant(quantizer_mode: str) -> Callable:
    """Given a quantizer mode, return the corresponding function for enabling fake
    quantize in a model prepared by the quantizer.
    If the quantizer is not recognized as a known QAT quantizer, we'll return None.
    """
    return _quantizer_mode_to_enable_fake_quant.get(quantizer_mode, None)
