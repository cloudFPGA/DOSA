from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.core.scaling import PowerOfTwoIntScaling
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.base import *


class Int5WeightPerTensorFixedPoint(
    NarrowIntQuant, MaxStatsScaling, PerTensorPoTScaling8bit, WeightQuantSolver):
    bit_width = 5


class Int5ActPerTensorFixedPoint(
    IntQuant, ParamFromRuntimePercentileScaling, PerTensorPoTScaling8bit, ActQuantSolver):
    bit_width = 5
