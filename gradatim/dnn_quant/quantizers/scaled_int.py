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
from brevitas.core.function_wrapper import TensorClamp
from brevitas.quant import Int8Bias, Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.quant.base import *
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.trunc import TruncQuantSolver


class Int32ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 32


class Int32WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width = 32


class Int32Bias(Int8Bias):
    bit_width = 32
    requires_input_bit_width = False


class Int16ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 16


class Int16WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width = 16


class Int16Bias(Int8Bias):
    bit_width = 16
    requires_input_bit_width = False


class Int7ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 7


class Int7WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width = 7


class Int7Bias(Int8Bias):
    bit_width = 7
    requires_input_bit_width = False


class Int6ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 6


class Int6WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width = 6


class Int6Bias(Int8Bias):
    bit_width = 6
    requires_input_bit_width = False


class Int5ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 5


class Int5WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width = 5


class Int5Bias(Int8Bias):
    bit_width = 5
    requires_input_bit_width = False


class Int4ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 4


class Int4WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width = 4


class Int4Bias(Int8Bias):
    bit_width = 4
    requires_input_bit_width = False


class Int3ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 3


class Int3WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width = 4


class Int3Bias(Int8Bias):
    bit_width = 3
    requires_input_bit_width = False
