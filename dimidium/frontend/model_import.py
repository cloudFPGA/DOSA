#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
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

from dimidium.frontend.TvmPrintMeta import PrintMeta
from dimidium.lib.dosa_dtype import get_bitwidth_of_DosaDtype
from dimidium.frontend.TvmGlobalTypeCast import CorrectionPipeline


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


def user_import(onnx_path, user_constraints, debug_mode=False):
    mod_i, params_i = onnx_import(onnx_path, user_constraints['shape_dict'],  repr(user_constraints['input_dtype']))
    print("\t...done.\n")

    if user_constraints['do_quantization']:
        print("DOSA: Executing TVM quantization...")
        mod_i, params_i = tvm_quantization(mod_i, params_i, user_constraints)
        print("\t...done.\n")
    # elif user_constraints['overwrite_imported_dtypes']:
    #     print("[DOSA:import:INFO] overwriting ONNX data types...")
    #     mod_i, params_i = overwrite_dtypes(mod_i, params_i, user_constraints)
    #     print("\t...done.\n")

    print("DOSA: Executing TVM optimization passes...")
    mod, params = tvm_optimization_pass(mod_i, params_i, debug=debug_mode)
    return mod, params

