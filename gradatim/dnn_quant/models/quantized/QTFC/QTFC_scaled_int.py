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
from .QTFC import QTFC
from gradatim.dnn_quant.quantizers import *


class QTFCInt32(QTFC):
    """
    Activations: int32, symmetric
    Weights: int32, symmetric,
    Bias: int32, symmetric
    """

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCInt32, self).__init__(hidden1, hidden2, hidden3,
                                        act_quant=Int32ActPerTensorFloat,
                                        weight_quant=Int32WeightPerTensorFloat,
                                        bias_quant=Int32Bias,
                                        bit_width=32)


class QTFCInt16(QTFC):
    """
    Activations: int16, symmetric
    Weights: int16, symmetric,
    Bias: int16, symmetric
    """

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCInt16, self).__init__(hidden1, hidden2, hidden3,
                                        act_quant=Int16ActPerTensorFloat,
                                        weight_quant=Int16WeightPerTensorFloat,
                                        bias_quant=Int16Bias,
                                        bit_width=16)


class QTFCInt8(QTFC):
    """
    Activations: int8, symmetric
    Weights: int8, symmetric,
    Bias: int8, symmetric
    """

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCInt8, self).__init__(hidden1, hidden2, hidden3,
                                       act_quant=Int8ActPerTensorFloat,
                                       weight_quant=Int8WeightPerTensorFloat,
                                       bias_quant=Int8Bias,
                                       bit_width=8)


class QTFCInt5(QTFC):
    """
    Activations: int5, symmetric
    Weights: int5, symmetric,
    Bias: int5, symmetric
    """

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCInt5, self).__init__(hidden1, hidden2, hidden3,
                                       act_quant=Int5ActPerTensorFloat,
                                       weight_quant=Int5WeightPerTensorFloat,
                                       bias_quant=Int5Bias,
                                       bit_width=5)


class QTFCInt4(QTFC):
    """
    Activations: int4, symmetric
    Weights: int4, symmetric,
    Bias: int4, symmetric
    """

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCInt4, self).__init__(hidden1, hidden2, hidden3,
                                       act_quant=Int4ActPerTensorFloat,
                                       weight_quant=Int4WeightPerTensorFloat,
                                       bias_quant=Int4Bias,
                                       bit_width=4)


class QTFCInt3(QTFC):
    """
    Activations: int3, symmetric
    Weights: int3, symmetric,
    Bias: int3, symmetric
    """

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCInt3, self).__init__(hidden1, hidden2, hidden3,
                                       act_quant=Int3ActPerTensorFloat,
                                       weight_quant=Int3WeightPerTensorFloat,
                                       bias_quant=Int3Bias,
                                       bit_width=3)
