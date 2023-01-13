import brevitas.nn as qnn
import onnx
import torch
import torch.nn as nn
from brevitas.quant import Int8WeightPerTensorFloat, Int8Bias, Int8ActPerTensorFloat

from src.data import export_data_as_numpy
from src.definitions import ROOT_DIR
from src.onnx import export_DOSA_onnx


class DummyDoubleLinear(nn.Module):
    def __init__(self):
        super(DummyDoubleLinear, self).__init__()
        self.qidd1 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(9, 5, bias=True,
                                  weight_quant=Int8WeightPerTensorFloat,
                                  bias_quant=Int8Bias,
                                  return_quant_tensor=False)
        self.qidd2 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(5, 3, bias=True,
                                  weight_quant=Int8WeightPerTensorFloat,
                                  bias_quant=Int8Bias,
                                  return_quant_tensor=False)

    def forward(self, x):
        x = self.qidd1(x)
        x = self.fc1(x)
        x = self.qidd2(x)
        x = self.fc2(x)
        return x


# define model
torch.manual_seed(0)
model = DummyDoubleLinear()

# prepare and save input
torch.manual_seed(0)
input = torch.rand((1, 9)) * 2 - 1
print('input: \n', input, '\n')
export_data_as_numpy('/home/sop/Documents/DNNQuantization/data/DummyDoubleLinear_input.npy', input,
                     data_transform=lambda x: model.qidd1(x).int())

# inference
model.eval()
model.fc1.cache_inference_quant_bias = True
model.fc2.cache_inference_quant_bias = True
print('quant result: \n', model(input), '\n')

model.cpu()
# bo.export_finn_onnx(model, (1, 9), ROOT_DIR + '/models/FINN/DummyLinear.onnx')
export_DOSA_onnx(model, (1, 9), ROOT_DIR+'/models/DOSA/DummyDoubleLinear.onnx')
