import torch
import torch.nn as nn


class StaticQuantModel(nn.Module):
    def __init__(self, model_fp32):
        super(StaticQuantModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.internal = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.internal(x)
        x = self.dequant(x)
        return x
