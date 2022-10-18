import torch
from brevitas.quant import Int8WeightPerTensorFloat
from torch.ao.quantization import QConfig, MinMaxObserver
from torch import nn
from brevitas import nn as qnn

from tests.torch_brevitas_comparisons.utils import prepare_torch_qlayer, prepare_brevitas_qlayer

custom_torch_config = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
)

# fp linear model
fp_linear = nn.Sequential(nn.Linear(5, 3, bias=False), nn.ReLU())
fp_linear.get_submodule('0').weight.data.fill_(0.0)
fp_linear.get_submodule('0').weight.data[0, 0] = 1.0

# calibration data
calibration_data = torch.zeros(1, 1, 5)

# torch quantized linear
torch_qlinear = prepare_torch_qlayer(fp_linear, custom_torch_config, calibration_data=calibration_data, fusion_list=['0', '1'])

# brevitas quantized linear
brevitas_qlinear = qnn.QuantLinear(5, 3, bias=False, weight_quant=Int8WeightPerTensorFloat, return_quant_tensor=True)
prepare_brevitas_qlayer(fp_linear.get_submodule('0'), brevitas_qlinear, calibration_data)

print(torch_qlinear.get_submodule('model_fp32.0').weight())
print(brevitas_qlinear.quant_weight())


