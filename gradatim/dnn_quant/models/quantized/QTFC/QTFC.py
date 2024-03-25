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
from brevitas import nn as qnn
from torch import nn

from gradatim.dnn_quant.models.quantized.quant_module import QuantModule
from gradatim.dnn_quant.utils import Reshape


class QTFC(QuantModule):
    """
    Base class for quantized TFC, per default not quantized and therefore acts as a wrapper for the full precision
    model
    """
    dropout = 0.2
    in_features = 28 * 28

    def __init__(self, hidden1, hidden2, hidden3,
                 act_quant=None,
                 weight_quant=None,
                 bias_quant=None,
                 bit_width=None):
        super(QTFC, self).__init__(num_act=4, num_weighted=4, num_biased=4)

        self.forward_step_index = 0

        a_quant, w_quant, b_quant, bit_width, return_qt, do_quantization =\
            self._process_quant_methods(act_quant, weight_quant, bias_quant, bit_width)

        # reshape layer
        self._append(Reshape(lambda x: (-1, QTFC.in_features)))

        self._append(qnn.QuantIdentity(act_quant=a_quant[0], return_quant_tensor=return_qt))
        self._append(nn.Dropout(p=QTFC.dropout))

        # first layer
        self._append(qnn.QuantLinear(QTFC.in_features, hidden1, bias=True, return_quant_tensor=False,
                                     weight_quant=w_quant[0], bias_quant=b_quant[0]))
        self._append(nn.BatchNorm1d(hidden1))
        self._append(qnn.QuantIdentity(act_quant=a_quant[1], return_quant_tensor=return_qt))
        self._append(nn.Dropout(p=QTFC.dropout))
        if do_quantization:
            self._append(qnn.QuantReLU(return_quant_tensor=return_qt, bit_width=bit_width))
        else:
            self._append(qnn.QuantReLU(act_quant=None))

        # second layer
        self._append(qnn.QuantLinear(hidden1, hidden2, bias=True, return_quant_tensor=False,
                                     weight_quant=w_quant[1], bias_quant=b_quant[1]))
        self._append(nn.BatchNorm1d(hidden2))
        self._append(qnn.QuantIdentity(act_quant=a_quant[2], return_quant_tensor=return_qt))
        self._append(nn.Dropout(p=QTFC.dropout))
        if do_quantization:
            self._append(qnn.QuantReLU(return_quant_tensor=return_qt, bit_width=bit_width))
        else:
            self._append(qnn.QuantReLU(act_quant=None))

        # third layer
        self._append(qnn.QuantLinear(hidden2, hidden3, bias=True, return_quant_tensor=False,
                                     weight_quant=w_quant[2], bias_quant=b_quant[2]))
        self._append(nn.BatchNorm1d(hidden3))
        self._append(qnn.QuantIdentity(act_quant=a_quant[3], return_quant_tensor=return_qt))
        self._append(nn.Dropout(p=QTFC.dropout))
        if do_quantization:
            self._append(qnn.QuantReLU(return_quant_tensor=return_qt, bit_width=bit_width))
        else:
            self._append(qnn.QuantReLU(act_quant=None))

        # output layer
        self._append(qnn.QuantLinear(hidden3, 10, bias=True, return_quant_tensor=False,
                                     weight_quant=w_quant[3], bias_quant=b_quant[3]))

    def forward(self, x):
        for module in self.features:
            x = module(x)
        return x

    def forward_step(self, x):
        if self.forward_step_index >= len(self.features):
            self.forward_step_index = 0
            return None, None, None

        module = self.features[self.forward_step_index]
        out = module(x)
        self.forward_step_index += 1
        return x, module, out
