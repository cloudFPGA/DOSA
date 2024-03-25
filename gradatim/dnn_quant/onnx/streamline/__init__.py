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

# Copyright (c) 2020, Xilinx
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
# * Neither the name of FINN nor the names of its
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
from qonnx.transformation.base import Transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.general import ConvertSubToAdd, ConvertDivToMul, GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps

from gradatim.dnn_quant.onnx.streamline.collapse_repeated import CollapseRepeatedAdd, CollapseRepeatedMul
from gradatim.dnn_quant.onnx.streamline.round_thresholds import RoundAndClipThresholds
from gradatim.dnn_quant.onnx.streamline.sign_to_thres import ConvertSignToThres

from .absorb import (
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
)

from .reorder import (
    MoveAddPastConv,
    MoveAddPastMul,
    MoveMulPastMaxPool,
    MoveScalarAddPastMatMul,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastConv,
    MoveScalarMulPastMatMul, MoveLinearPastEltwiseAdd, MoveLinearPastFork, MoveMaxPoolPastMultiThreshold,
)


class Streamline(Transformation):
    """Apply the streamlining transform, see arXiv:1709.04060."""

    def __init__(self, clean_step):
        self.clean_step = clean_step

    def apply(self, model):
        streamline_transformations = [
            ConvertSubToAdd(),
            ConvertDivToMul(),
            RemoveIdentityOps(),
            BatchNormToAffine(),
            ConvertSignToThres(),
            AbsorbSignBiasIntoMultiThreshold(),
            MoveMulPastMaxPool(),
            MoveScalarLinearPastInvariants(),
            MoveAddPastMul(),
            MoveScalarAddPastMatMul(),
            MoveAddPastConv(),
            MoveScalarMulPastMatMul(),
            MoveScalarMulPastConv(),
            MoveAddPastMul(),
            CollapseRepeatedAdd(),
            CollapseRepeatedMul(),
            MoveMulPastMaxPool(),
            AbsorbAddIntoMultiThreshold(),
            FactorOutMulSignMagnitude(),
            MoveMaxPoolPastMultiThreshold(),
            AbsorbMulIntoMultiThreshold(),
            Absorb1BitMulIntoMatMul(),
            Absorb1BitMulIntoConv(),
            RoundAndClipThresholds(),
            ConvertBipolarMatMulToXnorPopcount(),
            MoveLinearPastEltwiseAdd(),
            MoveLinearPastFork(),
            absorb.AbsorbScalarMulAddIntoTopK()
        ]
        for trn in streamline_transformations:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            if self.clean_step:
                model = model.transform(RemoveIdentityOps())
                model = model.transform(GiveReadableTensorNames())
                model = model.transform(InferDataTypes())
        return model, False
