import brevitas.nn as qnn
import onnx
import torch
from brevitas.quant import Int8WeightPerTensorFloat, Int8Bias, Int8ActPerTensorFloat
from torch import nn
import brevitas.onnx as bo

from src.data import export_data_as_numpy
from src.definitions import ROOT_DIR


class DummyLinear(nn.Module):
    def __init__(self):
        super(DummyLinear, self).__init__()
        self.quantidd = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.fc = qnn.QuantLinear(9, 3, bias=True,
                                  weight_quant=Int8WeightPerTensorFloat,
                                  bias_quant=Int8Bias,
                                  return_quant_tensor=True)

    def forward(self, x):
        x = self.quantidd(x)
        x = self.fc(x)
        return x


torch.manual_seed(0)
input = torch.rand((1, 9)) * 2 - 1
print('input: \n', input, '\n')
export_data_as_numpy('/home/sop/Documents/deployments/dummy-linear/driver/input.npy', input,
                     data_transform=lambda x: torch.floor(x * 128))

model = DummyLinear()
model.eval()
model.fc.cache_inference_quant_bias = True
print('quant result: \n', model(input), '\n')

model.cpu()
bo.export_finn_onnx(model, (1, 9), ROOT_DIR + '/models/DummyLinear.onnx')

# check onnx model
model_check = onnx.load(ROOT_DIR + '/models/DummyLinear.onnx')
onnx.checker.check_model(model_check)

# ========== Compare with unquantized inference result ==========
weights = model.fc.weight
bias = model.fc.bias
linear_layer = nn.Linear(9, 3, bias=True)
linear_layer.load_state_dict(model.fc.state_dict())
print('real result: \n', linear_layer(input), '\n')

# ========== step-by-step manual emulation (no brevitas) based on finn onnx description ============
e_scale = model.quantidd.quant_output_scale().item()  # 1/128
e_input = torch.floor(input / e_scale)  # <--- THIS is the input to the matrix multiplication node

w_scale = model.fc.quant_weight_scale().item()  # 0.0025
weights = torch.round(model.fc.weight / w_scale).transpose(0, 1)
linear_mul = torch.matmul(e_input, weights)

# They are not adding bias when running on Alveo !! And the onnx describes bias addition in fp domain
tot_scale = e_scale * w_scale  # 0.000019589
linear_mul_fp = linear_mul * tot_scale

# linear: add bias
b_scale = model.fc.quant_bias_scale()  # 0.000019589 = e_scale * w_scale
b_quant = model.fc.int_bias()
dequant_bias = (b_scale * b_quant)  # match values in onnx
real_bias = model.fc.bias

res = linear_mul_fp + dequant_bias
# res = linear_mul_fp + real_bias
print('personal emulation result\n', res, '\n')
print('\n')
