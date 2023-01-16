import torch
import onnx
from brevitas.export import export_finn_onnx

from src.data import export_data_as_npz
from src.definitions import ROOT_DIR
from src import data_loader, test
from src.models.full_precision import ResNet18
from src.models.quantized import QResNet18Int4
from src.onnx import export_DOSA_onnx
from tests.torch_brevitas_comparisons.utils import prepare_brevitas_qmodel


# Prepare CIFAR10 dataset
test_loader_cifar10 = data_loader(data_dir=ROOT_DIR + '/data', dataset='CIFAR10', batch_size=128, test=True, seed=0)
calibration_loader_cifar10, _ = data_loader(data_dir=ROOT_DIR + '/data', dataset='CIFAR10',
                                            batch_size=1, test=False, seed=0)

fp_model = ResNet18()
fp_model.load_state_dict(torch.load(ROOT_DIR+'/models/ResNet18.pt', map_location=torch.device('cpu')))

q_model = QResNet18Int4()
prepare_brevitas_qmodel(fp_model, q_model, data_loader=calibration_loader_cifar10, num_steps=300, seed=0)
print(q_model.get_quant_description((1, 3, 32, 32)))

# test model
test(q_model, test_loader_cifar10, seed=0)

# export onnx
q_model.cpu()
# export_finn_onnx(q_model, (1, 3, 32, 32), ROOT_DIR+'/models/FINN/QResNet18Int4.onnx')
export_DOSA_onnx(q_model, (1, 3, 32, 32), ROOT_DIR+'/models/DOSA/QResNet18Int4.onnx')

# check onnx model
# model = onnx.load(ROOT_DIR+'/models/FINN/QResNet18Int4.onnx')
model = onnx.load(ROOT_DIR+'/models/DOSA/QResNet18Int4.onnx')
onnx.checker.check_model(model)

# Export data used to test the model accuracy
export_data_as_npz(ROOT_DIR + '/data/cifar_test_data.npz', test_loader_cifar10, num_batches=None,
                   feature_transform=lambda x: q_model.features[0](x).int(), dtype='int8', seed=0)
