import brevitas.nn as qnn
import onnx
import torch
import numpy as np
from brevitas.quant import Int8WeightPerTensorFloat, Int8Bias, Int8ActPerTensorFloat
from torch import nn
import brevitas.onnx as bo

from src.data import export_data_as_numpy
from src.definitions import ROOT_DIR
from src.onnx import export_DOSA_onnx


class DummyDoubleConvolutional(nn.Module):
    def __init__(self):
        super(DummyDoubleConvolutional, self).__init__()
        self.qidd1 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(1, 1, 3, bias=True,
                                    weight_quant=Int8WeightPerTensorFloat,
                                    bias_quant=Int8Bias,
                                    return_quant_tensor=False)
        self.qidd2 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(1, 1, 2, bias=True,
                                     weight_quant=Int8WeightPerTensorFloat,
                                     bias_quant=Int8Bias,
                                     return_quant_tensor=False)

    def forward(self, x):
        x = self.qidd1(x)
        x = self.conv1(x)
        x = self.qidd2(x)
        x = self.conv2(x)
        return x

# define model
torch.manual_seed(0)
model = DummyDoubleConvolutional()

# prepare and save input
torch.manual_seed(0)
input = torch.rand((1, 1, 6, 6)) * 2 - 1
print('input: \n', input, '\n')
export_data_as_numpy('/home/sop/Documents/DNNQuantization/data/DummyDoubleConvolutional_quantized_input.npy', input,
                     data_transform=lambda x: model.qidd1(x).int())

# inference
model.eval()
model.conv1.cache_inference_quant_bias = True
model.conv2.cache_inference_quant_bias = True
print('quant result: \n', model(input), '\n')

# export
model.cpu()
bo.export_finn_onnx(model, (1, 1, 6, 6), ROOT_DIR + '/models/FINN/DummyDoubleConvolutional.onnx')
export_DOSA_onnx(model, (1, 1, 6, 6), ROOT_DIR + '/models/DOSA/DummyDoubleConvolutional.onnx')

# check onnx model
model_check = onnx.load(ROOT_DIR + '/models/FINN/DummyDoubleConvolutional.onnx')
model_check = onnx.load(ROOT_DIR + '/models/DOSA/DummyDoubleConvolutional.onnx')
onnx.checker.check_model(model_check)
