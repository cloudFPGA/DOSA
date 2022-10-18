import copy

import torch
from torch import nn

from src import calibrate
from src.models.quantized import QTFC
from src.test import controlled_calibrate


class StaticQuantModel(nn.Module):
    def __init__(self, model_fp32):
        super(StaticQuantModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model_fp32 = model_fp32
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x


def prepare_torch_qlayer(fp_model, qconfig, data_loader=None, calibration_data=None, fusion_list=None):
    fp_model.eval()
    model_fp_fused = copy.deepcopy(fp_model)

    if fusion_list is not None:
        internal_fused_model = torch.quantization.fuse_modules(
            model_fp_fused,
            fusion_list,
            inplace=True)
        torch_qmodel = StaticQuantModel(internal_fused_model)
    else:
        torch_qmodel = StaticQuantModel(model_fp_fused)
    torch_qmodel.qconfig = qconfig
    torch.ao.quantization.prepare(torch_qmodel, inplace=True)

    if calibration_data is None:
        calibrate(torch_qmodel, data_loader, num_steps=1, seed=42)
    else:
        controlled_calibrate(torch_qmodel, calibration_data)

    torch.ao.quantization.convert(torch_qmodel, inplace=True)
    return torch_qmodel


def prepare_brevitas_qmodel(fp_model, brevitas_model, data_loader=None, calibration_data=None):
    brevitas_model.load_model_state_dict(fp_model)

    if calibration_data is None:
        calibrate(brevitas_model, data_loader, num_steps=1, seed=42)
    else:
        controlled_calibrate(brevitas_model, calibration_data)


def prepare_brevitas_qlayer(fp_layer, brev_layer, calibration_data):
    brev_layer.load_state_dict(fp_layer.state_dict())
    controlled_calibrate(brev_layer, calibration_data)
