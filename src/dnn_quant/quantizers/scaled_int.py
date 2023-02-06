from brevitas.core.function_wrapper import TensorClamp
from brevitas.quant import Int8Bias, Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.quant.base import *
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.trunc import TruncQuantSolver


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
