from src.data import data_loader
from src.test import test
from src.models.full_precision import TFC
from src.models.quantized import QTFC

import torch


# Try to obtain the exact same results for pytorch and brevitas int8 quantization


def prepare_torch_qmodel(model_fp, data_loader):
    from tests.utils.torch_static_quant_model import StaticQuantModel
    import copy
    from src.test import calibrate

    model_fp.eval()
    model_fp_fused = copy.deepcopy(model_fp)
    internal_fused_model = torch.quantization.fuse_modules(
        model_fp_fused,
        [['1', '2'], ['5', '6'], ['9', '10']],
        inplace=True)
    torch_qmodel = StaticQuantModel(internal_fused_model)
    torch_qmodel.qconfig = torch.ao.quantization.default_qconfig
    torch.ao.quantization.prepare(torch_qmodel, inplace=True)
    calibrate(torch_qmodel, data_loader, 1)
    torch.ao.quantization.convert(torch_qmodel, inplace=True)
    return torch_qmodel


test_loader_mnist = data_loader(data_dir='../data', dataset='MNIST', batch_size=100, test=True)
model = TFC(64, 64, 64)
model.load_state_dict(torch.load('../models/TFC.pt', map_location=torch.device('cpu')))

# torch_qmodel = prepare_torch_qmodel(model, test_loader_mnist)
# for module in torch_qmodel.modules():
#     print(module)
#     print()

#test(torch_qmodel, test_loader_mnist)



brevitas_qmodel = QTFC(64, 64, 64)
brevitas_qmodel.load_model_state_dict(model)
test(brevitas_qmodel, test_loader_mnist, calibration_steps=100)
