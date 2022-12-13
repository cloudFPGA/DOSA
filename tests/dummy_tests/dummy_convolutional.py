import brevitas.nn as qnn
import onnx
import torch
import numpy as np
from brevitas.quant import Int8WeightPerTensorFloat, Int8Bias, Int8ActPerTensorFloat
from torch import nn
import brevitas.onnx as bo

from src.data import export_data_as_numpy
from src.definitions import ROOT_DIR


class DummyConvolutional(nn.Module):
    def __init__(self):
        super(DummyConvolutional, self).__init__()
        self.quantidd = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(1, 1, 2, bias=True,
                                    weight_quant=Int8WeightPerTensorFloat,
                                    bias_quant=Int8Bias,
                                    return_quant_tensor=False)

    def forward(self, x):
        x = self.quantidd(x)
        x = self.conv(x)
        return x


torch.manual_seed(0)
input = torch.rand((1, 1, 4, 4)) * 2 - 1
print('input: \n', input, '\n')
export_data_as_numpy('/home/sop/Documents/deployments/dummy-convolutional/driver/input.npy', input,
                     data_transform=lambda x: torch.floor(x * 128))

model = DummyConvolutional()
model.eval()
model.conv.cache_inference_quant_bias = True
print('quant result: \n', model(input), '\n')

model.cpu()
bo.export_finn_onnx(model, (1, 1, 4, 4), ROOT_DIR + '/models/DummyConvolutional.onnx')

# check onnx model
model_check = onnx.load(ROOT_DIR + '/models/DummyConvolutional.onnx')
onnx.checker.check_model(model_check)

# ========== Compare with unquantized convolution inference ==========
weights = model.conv.weight
bias = model.conv.bias
conv_layer = nn.Conv2d(1, 1, 2, bias=True)
conv_layer.load_state_dict(model.conv.state_dict())
print('real result: \n', conv_layer(input), '\n')

# ========== step-by-step manual emulation (no brevitas) based on finn onnx description ============
e_scale = model.quantidd.quant_output_scale().item()  # 1/128
e_input = torch.floor(input / e_scale)  # quantidentity
e_input = e_input.permute((0, 2, 3, 1))  # not actually necessary

# Finn "convolution input generator"
window_indices = np.arange(0, 4 * 4).reshape(-1, 4)
window_view = np.lib.stride_tricks.sliding_window_view(window_indices, (2, 2))[::1]
window_view = window_view.reshape((window_view.shape[0], window_view.shape[1], -1))
window_view = np.expand_dims(window_view, axis=0)
conv_input = e_input.flatten()[window_view]
conv_input = np.expand_dims(conv_input, axis=0)
conv_input = torch.tensor(conv_input)

# convolution with weights
w_scale = model.conv.quant_weight_scale().item()
weights = torch.floor(model.conv.weight / w_scale).reshape(4, -1)
conv_mul = torch.matmul(conv_input, weights).permute((0, 3, 1, 2))

# un-quantize first (this is cheating)
conv_scale = w_scale * e_scale
conv_mul = conv_mul * conv_scale

# convolution : add bias
b_scale = model.conv.quant_bias_scale()
b_quant = model.conv.int_bias()
dequant_bias = (b_scale * b_quant).item()  # same as in onnx: 0.0026482
real_bias = model.conv.bias

conv = conv_mul + dequant_bias
# conv = conv_mul + real_bias
print('personal emulation result\n', conv, '\n')
print('\n')
