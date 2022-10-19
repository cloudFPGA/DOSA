from brevitas.inject.enum import QuantType, BitWidthImplType, FloatToIntImplType, ScalingImplType, StatsOp, \
    RestrictValueType
from brevitas.quant.solver import ActQuantSolver

from src.models.quantized import QTFC, QTFCAffineQuantAct
from src.test import test
from src.data import data_loader
from src.model_processing import FullPrecisionModelIterator
from src.models.full_precision import TFC

import torch
from torch.quantization.qconfig import QConfig
from torch.quantization.observer import MinMaxObserver

from tests.torch_brevitas_comparisons.utils import prepare_torch_qlayer, prepare_brevitas_qmodel

# Try to obtain the exact same results for pytorch and brevitas int8 quantization

custom_torch_config = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
)


# ======= Main =======
test_loader_mnist = data_loader(data_dir='../../../data', dataset='MNIST', batch_size=100, test=True)
calibration_loader_mnist, _ = data_loader(data_dir='../../../data', dataset='MNIST', batch_size=1, test=False)

fp_model = TFC(64, 64, 64)
fp_model.load_state_dict(torch.load('../../models/TFC.pt', map_location=torch.device('cpu')))

# force bias to zero
mod_it = FullPrecisionModelIterator(fp_model)
mod_it.force_bias_zero()
fp_model.eval()

# full precision accuracy
# test(fp_model, test_loader_mnist)

calibration_data = torch.zeros(1, 1, 28, 28)
calibration_data[0, 0, 0, 1] = 1.0

# torch QuantIdentity scale and zero-point
fusion_list = [['1', '2'], ['5', '6'], ['9', '10']]
torch_quant_model = prepare_torch_qlayer(fp_model, custom_torch_config, data_loader=calibration_loader_mnist,
                                         calibration_data=None, fusion_list=fusion_list)
print("Torch quantized TFC:")
test(torch_quant_model, test_loader_mnist)

# brevitas QuantIdentity scale and zero-point
brevitas_quant_model = QTFC(64, 64, 64)
prepare_brevitas_qmodel(fp_model, brevitas_quant_model, data_loader=calibration_loader_mnist,
                        calibration_data=None)

print("Brevitas quantized TFC:")
test(brevitas_quant_model, test_loader_mnist)

print()
print("======= torch model =======")
print(torch_quant_model.get_submodule(''))

print("======= brevitas model =======")
print(brevitas_quant_model.get_quant_description())

# ======= Compare models weights (values, and scale factor and zero-point) =======
# Conclusion: the weights are quantized differently
print('====== Brevitas linear weights comparison ======')
for torch_i, brevitas_linear in zip(['1', '5', '9', '13'], ['2', '7', '12', '17']):
    torch_linear = torch_quant_model.get_submodule('model_fp32.' + torch_i)
    brevitas_linear = brevitas_quant_model.get_submodule('features.' + brevitas_linear)
    weight = torch_linear.weight()
    print('PyTorch: weight scale={}, weight zero-point={}'.format(torch_linear.weight().q_scale(),
                                                                  torch_linear.weight().q_zero_point()))
    print('Brevitas: weight scale={}, weight zero-point={}\n'.format(brevitas_linear.quant_weight().scale,
                                                                     brevitas_linear.quant_weight().zero_point))
