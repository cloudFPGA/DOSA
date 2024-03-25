#  /*******************************************************************************
#   * Copyright 2022 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#
from typing import Optional, Tuple, Union
from torch import Tensor
from qonnx.core.modelwrapper import ModelWrapper
from torch.nn import Module
from brevitas.quant_tensor import QuantTensor
import brevitas.onnx as bo

from gradatim.dnn_quant.onnx.fix_torch_export import fix_shared_initializers, fix_missing_opsets


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
