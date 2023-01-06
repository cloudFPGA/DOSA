import os
from typing import Optional, Tuple, Union

from brevitas.quant_tensor import QuantTensor
from qonnx.core.modelwrapper import ModelWrapper
from src.onnx.export_dataflow_steps import step_tidy_up, step_streamline
from torch import Tensor
from torch.nn import Module
import brevitas.onnx as bo


def export_DOSA_onnx(module: Module,
                     input_shape: Optional[Tuple[int, ...]] = None,
                     export_dir_path: Optional[str] = None,
                     input_t: Optional[Union[Tensor, QuantTensor]] = None,
                     **kwargs):

    model_file_prefix = export_dir_path + '/' + export_dir_path.split('/')[-1]
    if not os.path.exists(export_dir_path):
        os.makedirs(export_dir_path)

    print('step brevitas export')
    brevitas_model_file = model_file_prefix + '_brevitas.onnx'
    bo.export_brevitas_onnx(module, input_shape, brevitas_model_file, input_t)
    model = ModelWrapper(brevitas_model_file)

    print('step tidy up')
    model = step_tidy_up(model)
    model.save(model_file_prefix + 'step_tidy_up.onnx')
    print('step streamline')
    model = step_streamline(model)
    model.save(model_file_prefix + 'step_streamline.onnx')
