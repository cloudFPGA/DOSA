#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
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


def onnx_import(onnx_path, shape_dict, debug=False):
    onnx_model = onnx.load(onnx_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
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

