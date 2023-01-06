import os
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from qonnx.core.modelwrapper import ModelWrapper
from brevitas.quant_tensor import QuantTensor
import brevitas.onnx as bo

from src.onnx.export_dataflow_steps import step_tidy_up, step_streamline, step_finn_to_DOSA


def intermediate_models_path(export_path, export_intermediate_files):
    if export_path is None or not export_intermediate_files:
        return None

    split_path = export_path.split('/')
    dir_path_prefix = '/'.join(split_path[:-1])
    model_name = split_path[-1][:split_path[-1].index('.onnx')]
    dir_path = dir_path_prefix + '/' + model_name
    model_file_prefix = dir_path + '/' + model_name

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return model_file_prefix


def export_DOSA_onnx(module: Module,
                     input_shape: Optional[Tuple[int, ...]] = None,
                     export_path: Optional[str] = None,
                     input_t: Optional[Union[Tensor, QuantTensor]] = None,
                     export_intermediate_models=True):
    export_intermediate_models = export_intermediate_models if export_path else False

    model_path_prefix = intermediate_models_path(export_path, export_intermediate_models)
    model = export_step_brevitas(module, model_path_prefix, input_shape, input_t)
    model = export_step_tidy_up(model, model_path_prefix)
    model = export_step_streamline(model, model_path_prefix)
    return export_step_finn_to_DOSA(model, export_path)


def export_step_brevitas(module, model_file_prefix, input_shape, input_t):
    print('step brevitas export')
    brevitas_model_file = (model_file_prefix if model_file_prefix else '') + '_brevitas.onnx'
    bo.export_finn_onnx(module, input_shape, brevitas_model_file, input_t)
    model = ModelWrapper(brevitas_model_file)
    if model_file_prefix is None:
        os.remove(brevitas_model_file)
    return model


def export_step_tidy_up(model, model_file_prefix):
    print('step tidy up')
    model = step_tidy_up(model)
    if model_file_prefix is not None:
        model.save(model_file_prefix + '_step_tidy_up.onnx')
    return model


def export_step_streamline(model, model_file_prefix):
    print('step streamline')
    model = step_streamline(model)
    if model_file_prefix is not None:
        model.save(model_file_prefix + '_step_streamline.onnx')
    return model


def export_step_finn_to_DOSA(model, export_path):
    print('step finn to DOSA')
    model = step_finn_to_DOSA(model)
    if export_path is not None:
        model.save(export_path)
    return model


