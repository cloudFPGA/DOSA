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
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint


class Int5WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 5


class Int5ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 5


class Int4WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 4


class Int4ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 4


class Int3WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 3


class Int3ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 3
