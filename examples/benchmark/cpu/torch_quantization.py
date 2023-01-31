import os
import sys
import copy
import torch
import torch.nn as nn

src_path = os.path.abspath('../../../')
sys.path.insert(0, src_path)

from src import calibrate


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


def prepare_int8_static_qmodel(fp_model, dataloader, modules_to_fuse):
    from torch.quantization.qconfig import QConfig
    from torch.quantization.observer import MinMaxObserver

    internal_model = copy.deepcopy(fp_model)
    internal_model = torch.quantization.fuse_modules(
        internal_model,
        modules_to_fuse,
        inplace=True
    )

    my_qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=True)
    )

    q_model = StaticQuantModel(internal_model)
    q_model.qconfig = my_qconfig
    torch.ao.quantization.prepare(q_model, inplace=True)
    calibrate(q_model, dataloader, num_steps=1, seed=42)
    torch.ao.quantization.convert(q_model, inplace=True)
    return q_model


def prepare_int8_dynamic_qmodel(fp_model, modules_to_quantize):
    return torch.quantization.quantize_dynamic(
        fp_model, modules_to_quantize, dtype=torch.qint8
    )
