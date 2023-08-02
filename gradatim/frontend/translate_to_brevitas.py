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
#  *     Created: Aug 2023
#  *     Authors: NGL
#  *
#  *     Description:
#  *        module to translate a torch script module to a brevitas module
#  *
#  *

from gradatim.dnn_quant.models.quantized import QuantModule


def translate_to_quantized_model(fp_model, bit_width):
    # TODO:
    #  1. count activations, weights and biases...for internal lists
    #  2. then, append layer by layer (_append())
    q_model = QuantModule(num_act=42)

    return q_model

