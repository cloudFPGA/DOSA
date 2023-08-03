import os
import shutil
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from qonnx.core.modelwrapper import ModelWrapper
from brevitas.quant_tensor import QuantTensor
import brevitas.onnx as bo

from gradatim.dnn_quant.onnx.export_dataflow_steps import step_tidy_up, step_streamline, step_finn_to_DOSA
from gradatim.dnn_quant.onnx.fix_torch_export import fix_missing_opsets, fix_shared_initializers


def delete_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Previous exported model deleted!")


def intermediate_models_path(export_path, export_intermediate_files):
    if export_path is None or not export_intermediate_files:
        return None

    split_path = export_path.split('/')
    dir_path_prefix = '/'.join(split_path[:-1])
    model_name = split_path[-1][:split_path[-1].index('.onnx')]
    dir_path = dir_path_prefix + '/' + model_name + '_intermediate_models'
    model_file_prefix = dir_path + '/' + model_name

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print("Previous exported models deleted!")
    os.makedirs(dir_path)

    return model_file_prefix


def export_DOSA_onnx(module: Module,
                     input_shape: Optional[Tuple[int, ...]] = None,
                     export_path: Optional[str] = None,
                     input_t: Optional[Union[Tensor, QuantTensor]] = None,
                     export_intermediate_models=True):
    export_intermediate_models = export_intermediate_models if export_path else False

    model_path_prefix = intermediate_models_path(export_path, export_intermediate_models)
    # the steps below process/modify the model even if the intermediate steps aren't saved
    #  SOP: "modifies and return a qonnx.core.ModelWrapper instance with all the onnx final nodes that you can save yourself to an onnx file."
    model = export_step_brevitas(module, model_path_prefix, input_shape, input_t)
    model = export_step_tidy_up(model, model_path_prefix)
    model = export_step_streamline(model, model_path_prefix)
    return export_step_finn_to_DOSA(model, export_path)


def export_step_brevitas(module, model_file_prefix, input_shape, input_t):
    print('step brevitas export')
    brevitas_model_file = (model_file_prefix if model_file_prefix else '') + '_brevitas.onnx'
    bo.export_finn_onnx(module=module, input_shape=input_shape, export_path=brevitas_model_file, input_t=input_t)
    model = ModelWrapper(brevitas_model_file)

    # overcome torch and brevitas export errors
    model = fix_missing_opsets(model)
    model = fix_shared_initializers(model)

    if model_file_prefix is None:
        os.remove(brevitas_model_file)
    return model


def export_step_tidy_up(model, model_file_prefix):
    print('step tidy up')
    model = step_tidy_up(model)
    model = fix_missing_opsets(model)
    if model_file_prefix is not None:
        model.save(model_file_prefix + '_step_tidy_up.onnx')
    return model


def export_step_streamline(model, model_file_prefix):
    print('step streamline')
    model = step_streamline(model)
    model = fix_missing_opsets(model)
    if model_file_prefix is not None:
        model.save(model_file_prefix + '_step_streamline.onnx')
    return model


def export_step_finn_to_DOSA(model, export_path):
    print('step finn to DOSA')
    model = step_finn_to_DOSA(model)
    model = fix_missing_opsets(model)
    if export_path is not None:
        model.save(export_path)
    return model


