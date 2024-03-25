#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
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
import os.path

#  *
#  *                       cloudFPGA
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *       Library for importing and optimizing the NN model
#  *
#  *

import onnx
import tvm
import tvm.relay as relay
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import gradatim.lib.singleton as dosa_singleton
from gradatim.frontend.TvmPrintMeta import PrintMeta
from gradatim.lib.dosa_dtype import get_bitwidth_of_DosaDtype
from gradatim.frontend.TvmGlobalTypeCast import CorrectionPipeline
from gradatim.frontend.translate_to_brevitas import translate_to_quantized_model
from gradatim.dnn_quant.onnx.export_DOSA import export_DOSA_onnx


def onnx_import(onnx_path, shape_dict, input_dtype, debug=False):
    onnx_model = onnx.load(onnx_path)
    # freeze_params=True is important, otherwise copy and certain visitors can't work
    # frozen parameters also lead to more optimizations and higher level operators
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype=input_dtype, freeze_params=True)
    if debug:
        print(mod.astext(show_meta_data=False))
    return mod, params


def tvm_optimization_pass(mod, params, debug=False):

    # first, TVM optimization pass
    seq1_calls = [
        relay.transform.FoldConstant(),
        relay.transform.FastMath(),
        relay.transform.CanonicalizeCast(),
        relay.transform.DeadCodeElimination(),
        relay.transform.FuseOps(),
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.EliminateCommonSubexpr(),
        # tvm.transform.PrintIR(),
        relay.transform.SimplifyInference(),
        relay.transform.FoldExplicitPadding(),
        relay.transform.ForwardFoldScaleAxis(),
        relay.transform.InferType(),
        # relay.transform.AnnotateSpans(),  # not working with input.1 name...
    ]

    pass_instruments = []
    if debug:
        pass_instruments.append(PrintMeta())
        seq1_calls.append(tvm.transform.PrintIR())

    seq1 = tvm.transform.Sequential(seq1_calls)

    with tvm.transform.PassContext(opt_level=3, instruments=pass_instruments):
        mod2 = seq1(mod)

    return mod2, params


def tvm_quantization(mod, params, user_constraints):
    input_size_t = user_constraints['used_input_size_t']
    input_dtype = repr(user_constraints['input_dtype'])
    weight_dtype = repr(user_constraints['target_dtype'])
    weight_size_t = get_bitwidth_of_DosaDtype(user_constraints['target_dtype'])

    with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0,
                                nbit_input=input_size_t, nbit_weight=weight_size_t, dtype_input=input_dtype,
                                dtype_weight=weight_dtype, dtype_activation=weight_dtype):
        mod = relay.quantize.quantize(mod)
    return mod, params


def overwrite_dtypes(mod, params, user_constraints):
    custom_pass = CorrectionPipeline(input_name=list(user_constraints['shape_dict'].keys())[0],
                                     var_type=user_constraints['overwrite_dtypes']['data'],
                                     constant_type=user_constraints['overwrite_dtypes']['weights'])
    mod_2 = custom_pass(mod)
    return mod_2, params


def user_import_from_onnx(onnx_path, user_constraints, debug_mode=False):
    mod_i, params_i = onnx_import(onnx_path, user_constraints['shape_dict'],  repr(user_constraints['input_dtype']))
    print("\t...done.\n")

    if user_constraints['do_quantization']:
        if user_constraints['use_tvm_quantization']:
            print("DOSA: Executing TVM quantization...")
            mod_i, params_i = tvm_quantization(mod_i, params_i, user_constraints)
            print("\t...done.\n")
        else:
            print('[DOSA:IMPORT:ERROR] quantization method not support within onnx flow. STOP.')
            exit(1)
    # elif user_constraints['overwrite_imported_dtypes']:
    #     print("[DOSA:import:INFO] overwriting ONNX data types...")
    #     mod_i, params_i = overwrite_dtypes(mod_i, params_i, user_constraints)
    #     print("\t...done.\n")

    print("DOSA: Executing TVM optimization passes...")
    mod, params = tvm_optimization_pass(mod_i, params_i, debug=debug_mode)
    return mod, params


def user_import_from_torchscript(model_path, user_constraints, calibration_data_path, debug_mode=False):
    dnn = torch.jit.load(model_path)
    dnn.eval()

    if user_constraints['do_quantization'] and calibration_data_path is not None:
        print("\t...done.\nDOSA: Starting quantization translation...")
        input_size_t = user_constraints['used_input_size_t']
        input_dtype = repr(user_constraints['input_dtype'])
        in_shape_tupel = tuple([v for k,v in user_constraints['shape_dict'].items()][0])
        weight_dtype = repr(user_constraints['target_dtype'])
        weight_size_t = get_bitwidth_of_DosaDtype(user_constraints['target_dtype'])

        q_model = translate_to_quantized_model(dnn, weight_size_t)
        # calibration
        calibration_data = np.load(calibration_data_path).astype('float32')  # TODO?
        ignore_labels = np.zeros(calibration_data.shape[0]).astype('float32')  # TODO?
        torch_dataset = TensorDataset(torch.from_numpy(calibration_data), torch.from_numpy(ignore_labels))
        torch_dataloader = DataLoader(torch_dataset)
        # num_steps = min(300, len(calibration_data))
        num_steps = len(calibration_data)
        print(f"\t...done.\nDOSA: Starting quantization calibration ({num_steps} steps) ...")
        q_model.load_state_and_calibrate(dnn, data_loader=torch_dataloader, num_steps=num_steps, seed=42)
        q_model.cpu()
        if debug_mode:
            print(f"\n----- {q_model.name} Description -----")
            print(q_model.get_quant_description(in_shape_tupel))
        # we cannot test for accuracy with just calibration data
        print("\t...done.\nDOSA: Building quantized AST...")
        # export to tmp-onnx, import from tmp-onnx (in build-dir/quantization?)
        onnx_folder_path = os.path.abspath(f"{dosa_singleton.config.global_build_dir}/quantized_model/")
        os.system(f"mkdir -p {onnx_folder_path}")
        onnx_model_path = f"{onnx_folder_path}/{q_model.name.replace(' ', '')}_quantized.onnx"
        list_of_removed_float_transformations = []
        export_DOSA_onnx(module=q_model, input_shape=in_shape_tupel, export_path=onnx_model_path,
                         returnlist_removed_inputs=list_of_removed_float_transformations)
        dosa_singleton.objects.quant_module.list_of_removed_float_transformations = list_of_removed_float_transformations
        new_input_dict = {'global_in': list(in_shape_tupel)}
        mod_i, params_i = onnx_import(onnx_model_path, new_input_dict, input_dtype, debug_mode)
        # overwrite datatypes later
        dosa_singleton.config.quant.overwrite_imported_dtypes = True
        # TODO: allow different dtypes?
        dosa_singleton.config.quant.activation_dtype = user_constraints['target_dtype']
        dosa_singleton.config.quant.weight_dtype = user_constraints['target_dtype']
        dosa_singleton.config.quant.bias_dtype = user_constraints['target_dtype']
        dosa_singleton.config.quant.numbers_already_scaled = True
        print("\t...done.\n")
    else:
        # import directly into tvm
        mod_i, params_i = relay.frontend.from_pytorch(dnn, user_constraints['shape_dict'], keep_quantized_weight=True)
        # freeze params?
        if debug_mode:
            print(mod_i.astext(show_meta_data=False))
        if user_constraints['do_quantization']:
            print("\t...done (Skipped quantization as requested).\n")
        else:
            print("\t...done (Skipped quantization due to MISSING CALIBRATION DATA).\n")

    print("DOSA: Executing TVM optimization passes...")
    mod, params = tvm_optimization_pass(mod_i, params_i, debug=debug_mode)
    return mod, params



