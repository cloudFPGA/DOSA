from brevitas.core.function_wrapper import TensorClamp
from brevitas.quant.base import *
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.trunc import TruncQuantSolver


class Int8ActPerTensorFloatMax(
    IntQuant, MaxStatsScaling, PerTensorFloatScaling8bit, ActQuantSolver):
    pass


