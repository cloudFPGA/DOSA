import brevitas.nn as qnn
import onnx
import torch
from brevitas.quant import Int8WeightPerTensorFloat, Int8Bias, Int8ActPerTensorFloat
from torch import nn
import brevitas.onnx as bo

from src.definitions import ROOT_DIR


class DummyLinear(nn.Module):
    def __init__(self):
        super(DummyLinear, self).__init__()

        self.quantidd = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

        self.fc = qnn.QuantLinear(5, 3, bias=True,
                                  weight_quant=Int8WeightPerTensorFloat,
                                  bias_quant=Int8Bias,
                                  return_quant_tensor=False)
        # self.quantidd2 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        # self.relu = qnn.QuantReLU(return_quant_tensor=False)

    def forward(self, x):
        x = self.quantidd(x)
        x = self.fc(x)
        # x = self.quantidd2(x)
        # x = self.relu(x)
        return x


torch.manual_seed(0)
input = torch.randn((1, 1, 5))
print(input)

model = DummyLinear()
model.eval()
print(model(input))

model.cpu()
bo.export_finn_onnx(model, (1, 1, 5), ROOT_DIR + '/models/DummyLinear.onnx')

# check onnx model
model = onnx.load(ROOT_DIR + '/models/DummyLinear.onnx')
onnx.checker.check_model(model)
