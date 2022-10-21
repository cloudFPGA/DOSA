from brevitas.core.function_wrapper import TensorClamp
from brevitas.quant import IntBias, Int8Bias
from brevitas.quant.base import *
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.trunc import TruncQuantSolver


class Int8ActPerTensorFloatMax(
    IntQuant, MaxStatsScaling, PerTensorFloatScaling8bit, ActQuantSolver):
    pass


class Int5ActPerTensorFloat(
    IntQuant, ParamFromRuntimePercentileScaling, PerTensorFloatScaling8bit, ActQuantSolver):
    bit_width = 5


class Int5WeightPerTensorFloat(
    NarrowIntQuant, MaxStatsScaling, PerTensorFloatScaling8bit, WeightQuantSolver):
    bit_width = 5


class Int5Bias(Int8Bias):
    bit_width = 5
    requires_input_bit_width = False
