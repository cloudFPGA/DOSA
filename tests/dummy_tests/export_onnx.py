import onnx
import torch
from torch import nn
from brevitas import nn as qnn
from brevitas.quant import Int8Bias, Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint
import brevitas.onnx as bo

from src.definitions import ROOT_DIR


class DummyLinear(nn.Module):
    def __init__(self):
        super(DummyLinear, self).__init__()

        self.features = nn.ModuleList()
        self.features.append(qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True))
        self.features.append(qnn.QuantLinear(10, 3, bias=True,
                                             weight_quant=Int8WeightPerTensorFixedPoint,
                                             bias_quant=Int8Bias))

    def forward(self, x):
        for model in self.features:
            x = model(x)
        return x


model = DummyLinear()

# calibrate
model.train()
torch.manual_seed(0)
for i in range(100):
    input = torch.randn((1, 10))
    model(input)
model.eval()

model.cpu()
quant_idd = qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)
input = quant_idd(torch.randn((1, 10)))
bo.export_finn_onnx(model, export_path=ROOT_DIR+'/models/DummyLinear.onnx', input_t=input)

# check onnx model
model = onnx.load(ROOT_DIR+'/models/DummyLinear.onnx')
onnx.checker.check_model(model)
