# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors, SortGraph,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.remove import RemoveIdentityOps

from gradatim.dnn_quant.onnx.finn_to_DOSA import RemoveFloatPointNodes
from gradatim.dnn_quant.onnx.finn_to_DOSA.thresholds import *
from gradatim.dnn_quant.onnx.streamline import Streamline
from gradatim.dnn_quant.onnx.streamline import *


def step_tidy_up(model: ModelWrapper):
    """Run the tidy-up step on given model. This includes shape and datatype
    inference, constant folding, and giving nodes and tensors better names.
    """

    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    return model


def step_streamline(model: ModelWrapper, clean_step=True):
    """Run streamlining on given model. Streamlining involves moving floating point
    scale/shift parameters around, collapsing adjacent ones into a single parameter,
    then absorbing the scale/shift into the following `MultiThreshold` node.
    Streamlining requires careful topology design and cannot be applied to all
    topologies.
    """
    model_prev = None

    # do it as long as there are changes
    while model.model != model_prev:
        model_prev = copy.deepcopy(model.model)
        model = model.transform(Streamline(clean_step))

        # big loop tidy up
        model = model.transform(RemoveIdentityOps())
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())

    return model


# TODO Sophie cleanup
# def step_streamline_linear(model: ModelWrapper):
#     streamline_transformations = [
#         # AbsorbScalarMulAddIntoTopK(),  # before MoveAddPastMul to avoid int->float
#         ConvertSubToAdd(),
#         ConvertDivToMul(),
#         RemoveIdentityOps(),
#         CollapseRepeatedMul(),
#         BatchNormToAffine(),
#         ConvertSignToThres(),
#         AbsorbSignBiasIntoMultiThreshold(),
#         MoveAddPastMul(),
#         MoveScalarAddPastMatMul(),
#         MoveAddPastConv(),
#         MoveScalarMulPastMatMul(),
#         MoveScalarMulPastConv(),
#         MoveScalarLinearPastInvariants(),
#         MoveAddPastMul(),
#         CollapseRepeatedAdd(),
#         CollapseRepeatedMul(),
#         AbsorbAddIntoMultiThreshold(),
#         FactorOutMulSignMagnitude(),
#         MoveMaxPoolPastMultiThreshold(),
#         AbsorbMulIntoMultiThreshold(),
#         Absorb1BitMulIntoMatMul(),
#         Absorb1BitMulIntoConv(),
#         RoundAndClipThresholds(),
#     ]
#     for trn in streamline_transformations:
#         model = model.transform(trn)
#         model = model.transform(GiveUniqueNodeNames())
#     return model
#
#
# def step_streamline_nonlinear(model: ModelWrapper):
#     streamline_transformations = [
#         MoveLinearPastEltwiseAdd(),
#         MoveLinearPastFork(),
#     ]
#     for trn in streamline_transformations:
#         model = model.transform(trn)
#         model = model.transform(GiveUniqueNodeNames())
#     return model
#
#
# def step_streamline(model: ModelWrapper):
#
#     for iter_id in range(4):
#         model = step_streamline_linear(model)
#         model = step_streamline_nonlinear(model)
#
#         # big loop tidy up
#         model = model.transform(RemoveUnusedTensors())
#         model = model.transform(GiveReadableTensorNames())
#         model = model.transform(InferDataTypes())
#         model = model.transform(SortGraph())
#
#     model = model.transform(DoubleToSingleFloat())
#
#     return model


def step_finn_to_DOSA(model: ModelWrapper, returnlist_removed_inputs=None):
    """This step transforms finn custom nodes found in the model to ONNX standard operators, as DOSA doesn't support
    custom operators. Note however that DOSA will interpret the operator

    This step will only execute if QONNX nodes are found.
    These include the following op_types: "Quant" , "Trunc" and "BinaryQuant".
    If such nodes are found the step will run the tidy-up step from QONNX
    and then convert the QONNX model to the FINN-ONNX dialect.
    """
    model = model.transform(ThresMissingOutBiasToZero())
    if returnlist_removed_inputs is not None:
        model = model.transform(RemoveFloatPointNodes(returnlist_removed_nodes=returnlist_removed_inputs))
    else:
        model = model.transform(RemoveFloatPointNodes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())

    return model
