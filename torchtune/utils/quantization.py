# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

# importing TORCH_VERSION_AFTER_2_3 because `Int8DynActInt4WeightQuantizer`
# is only available after 2.3 so we have to guard the pytorch versions to decide
# the list of supported quantizers
from torchao.utils import TORCH_VERSION_AFTER_2_3, TORCH_VERSION_AFTER_2_4

__all__ = [
    "get_quantizer_mode",
]


_quantizer_to_mode = {}
_quantizer_mode_to_disable_fake_quant = {}
_quantizer_mode_to_enable_fake_quant = {}


if TORCH_VERSION_AFTER_2_3:
    from torchao.quantization.quant_api import (
        quantize_,
        int4_weight_only,
        int8_dynamic_activation_int4_weight,
        TensorCoreTiledLayoutType,
        #Int8DynActInt4WeightQuantizer,
        #Int4WeightOnlyQuantizer,
    )

    class Int8DynActInt4WeightQuantizer:
        def __init__(self, groupsize: int=256):
            self.groupsize = groupsize

        def quantize(self, model):
            quantize_fn = int8_dynamic_activation_int4_weight(self.groupsize)
            quantize_(model, quantize_fn)
            return model

    class Int4WeightOnlyQuantizer:
        def __init__(self, groupsize: int=256, inner_k_tiles: int=8):
            self.groupsize = groupsize
            self.inner_k_tiles = inner_k_tiles

        def quantize(self, model):
            layout_type = TensorCoreTiledLayoutType(self.inner_k_tiles)
            quantize_fn = int4_weight_only(self.groupsize, layout_type)
            quantize_(model, quantize_fn)
            return model

    __all__.append("Int8DynActInt4WeightQuantizer")
    __all__.append("Int4WeightOnlyQuantizer")
    _quantizer_to_mode[Int8DynActInt4WeightQuantizer] = "8da4w"
    _quantizer_to_mode[Int4WeightOnlyQuantizer] = "4w"


if TORCH_VERSION_AFTER_2_4:
    from torchao.quantization.prototype.qat import (
        disable_4w_fake_quant,
        disable_8da4w_fake_quant,
        enable_4w_fake_quant,
        enable_8da4w_fake_quant,
        Int4WeightOnlyQATQuantizer,
        Int8DynActInt4WeightQATQuantizer,
    )

    __all__.append("Int8DynActInt4WeightQATQuantizer")
    __all__.append("Int4WeightOnlyQATQuantizer")
    _quantizer_to_mode[Int8DynActInt4WeightQATQuantizer] = "8da4w-qat"
    _quantizer_to_mode[Int4WeightOnlyQATQuantizer] = "4w-qat"
    _quantizer_mode_to_disable_fake_quant["8da4w-qat"] = disable_8da4w_fake_quant
    _quantizer_mode_to_enable_fake_quant["8da4w-qat"] = enable_8da4w_fake_quant
    _quantizer_mode_to_disable_fake_quant["4w-qat"] = disable_4w_fake_quant
    _quantizer_mode_to_enable_fake_quant["4w-qat"] = enable_4w_fake_quant


def get_quantizer_mode(quantizer: Optional[Callable]) -> Optional[str]:
    """Given a quantizer object, returns a string that specifies the type of quantization.

    For example, in the case of int4 weight only quantization, we'll return "4w".
    If the quantizer is not recognized as a known quantizer, we'll return None.

    Currently supported:

    - :class:`~torchao.quantization.quant_api.Int8DynActInt4WeightQuantizer`: "8da4w" (requires ``torch>=2.3.0``)
    - :class:`~torchao.quantization.prototype.qat.Int8DynActInt4WeightQATQuantizer`: "8da4w-qat" (requires ``torch>=2.4.0``)

    Args:
        quantizer (Optional[Callable]): A callable object that implements the `quantize` method.

    Returns:
        Optional[str]: The quantization mode.
    """
    return _quantizer_to_mode.get(type(quantizer), None)


def _get_disable_fake_quant(quantizer_mode: str) -> Callable:
    """Given a quantizer mode, return the corresponding function for disabling fake
    quantize in a model prepared by the quantizer.
    If the quantizer is not recognized as a known QAT quantizer, return None.
    """
    return _quantizer_mode_to_disable_fake_quant.get(quantizer_mode, None)


def _get_enable_fake_quant(quantizer_mode: str) -> Callable:
    """Given a quantizer mode, return the corresponding function for enabling fake
    quantize in a model prepared by the quantizer.
    If the quantizer is not recognized as a known QAT quantizer, return None.
    """
    return _quantizer_mode_to_enable_fake_quant.get(quantizer_mode, None)
