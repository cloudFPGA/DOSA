from typing import Optional, Tuple, Union
from torch import Tensor
from qonnx.core.modelwrapper import ModelWrapper
from torch.nn import Module
from brevitas.quant_tensor import QuantTensor
import brevitas.onnx as bo

from dnn_quant.onnx.fix_torch_export import fix_shared_initializers, fix_missing_opsets


def export_FINN_onnx(module: Module,
                     input_shape: Optional[Tuple[int, ...]] = None,
                     export_path: Optional[str] = None,
                     input_t: Optional[Union[Tensor, QuantTensor]] = None):
    model = bo.export_finn_onnx(module=module, input_shape=input_shape, input_t=input_t)

    model = ModelWrapper(model)
    model = fix_missing_opsets(model)
    model = fix_shared_initializers(model)
    model.save(export_path)
    return model
