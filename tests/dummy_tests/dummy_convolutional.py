import brevitas.nn as qnn
import onnx
import torch
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
        # self.quantidd2 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        # self.relu = qnn.QuantReLU(return_quant_tensor=False)

    def forward(self, x):
        x = self.quantidd(x)
        x = self.conv(x)
        # x = self.quantidd2(x)
        # x = self.relu(x)
        return x


torch.manual_seed(0)
input = torch.randn((1, 1, 4, 4))
print(input)
# export_data_as_numpy('/home/sop/Documents/deployments/dummy-convolutional/driver/input.npy', input,
#                      data_transform=lambda x: torch.floor(x * 127))

model = DummyConvolutional()
model.eval()
print(model(input))

model.cpu()
bo.export_finn_onnx(model, (1, 1, 4, 4), ROOT_DIR+'/models/DummyConvolutional.onnx')

# check onnx model
model = onnx.load(ROOT_DIR+'/models/DummyConvolutional.onnx')
onnx.checker.check_model(model)

## step-by-step manual emulation (no brevitas)
