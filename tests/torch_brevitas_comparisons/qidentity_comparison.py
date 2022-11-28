from brevitas.core.bit_width import BitWidthConst
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.quant import RescalingIntQuant, IntQuant
from brevitas.core.scaling import RuntimeStatsScaling, IntScaling
from brevitas.core.stats import AbsMinMax
from brevitas.core.zero_point import ZeroZeroPoint, UIntSymmetricZeroPoint
from brevitas.inject.enum import QuantType, BitWidthImplType, FloatToIntImplType, ScalingImplType, StatsOp, \
    RestrictValueType
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant.base import UintQuant, ParamFromRuntimeMinMaxScaling, PerTensorFloatScaling8bit, \
    ParamFromRuntimePercentileScaling
from brevitas.quant.solver import ActQuantSolver

from src.data import data_loader
from src.definitions import ROOT_DIR
from src.test import test, calibrate
from src.models.full_precision import TFC

from brevitas.inject import ExtendedInjector
from brevitas import nn as qnn

import torch
from torch import nn
from torch.quantization.qconfig import QConfig
from torch.quantization.observer import MinMaxObserver
from tests.torch_brevitas_comparisons.utils import StaticQuantModel
import copy

# Try to obtain the exact same results for pytorch and brevitas int8 quantization

custom_torch_config = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=True)
)


class CustomBrevitasConfig(ActQuantSolver):
    # quant type
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False

    # zero-point implementation
    zero_point_impl = UIntSymmetricZeroPoint

    # scaling implementation
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MIN_MAX
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class QuantStubWrapperModel(nn.Module):
    def __init__(self, quant_stub):
        super(QuantStubWrapperModel, self).__init__()
        self.quant_stub = quant_stub

    def forward(self, x):
        return self.quant_stub(x)


def prepare_torch_qmodel(model_fp, data_loader):
    model_fp.eval()
    model_fp_fused = copy.deepcopy(model_fp)
    internal_fused_model = torch.quantization.fuse_modules(
        model_fp_fused,
        [['1', '2'], ['5', '6'], ['9', '10']],
        inplace=True)
    torch_qmodel = StaticQuantModel(internal_fused_model)
    torch_qmodel.qconfig = custom_torch_config
    torch.ao.quantization.prepare(torch_qmodel, inplace=True)
    calibrate(torch_qmodel, data_loader, num_steps=1, seed=42)
    torch.ao.quantization.convert(torch_qmodel, inplace=True)
    return torch_qmodel


def prepare_simple_torch_quant_identity(data_loader):
    quant_idd = QuantStubWrapperModel(torch.quantization.QuantStub())
    quant_idd.qconfig = custom_torch_config
    torch.ao.quantization.prepare(quant_idd, inplace=True)
    calibrate(quant_idd, data_loader, num_steps=1, seed=42)
    torch.ao.quantization.convert(quant_idd, inplace=True)
    return quant_idd


def prepare_simple_brevitas_quant_identity(data_loader):
    quant_idd = qnn.QuantIdentity(act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
    calibrate(quant_idd, data_loader, num_steps=1, seed=42)
    quant_idd.eval()
    return quant_idd


# ======= Main =======
test_loader_mnist = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=1, test=True)

torch.manual_seed(42)
calibration_data = next(iter(test_loader_mnist))[0]
print('Calibration data: min={}, max={}\n'.format(calibration_data.min(), calibration_data.max()))

# torch QuantIdentity scale and zero-point
torch_quant_idd = prepare_simple_torch_quant_identity(test_loader_mnist)
torch_scale = torch_quant_idd.quant_stub.scale
torch_zero_point = torch_quant_idd.quant_stub.zero_point
print('Torch QuantIdentity computed: scale={}, zero-point={}'.format(torch_scale, torch_zero_point))

# brevitas QuantIdentity scale and zero-point
brevitas_quant_idd = prepare_simple_brevitas_quant_identity(test_loader_mnist)
brevitas_scale = brevitas_quant_idd(calibration_data).scale
brevitas_zero_point = brevitas_quant_idd(calibration_data).zero_point
print('Brevitas QuantIdentity computed: scale={}, zero-point={}'.format(brevitas_scale, brevitas_zero_point))
