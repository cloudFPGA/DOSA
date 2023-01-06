import torch
import onnx
from brevitas.export import export_finn_onnx

from src.data import export_data_as_npz
from src.definitions import ROOT_DIR
from src import data_loader, test
from src.module_processing import FullPrecisionModuleIterator
from src.models.full_precision.TFC import TFC
from src.models.quantized import QTFCInt8
from src.onnx import export_DOSA_onnx
from tests.torch_brevitas_comparisons.utils import prepare_brevitas_qmodel


# Prepare MNIST dataset
test_loader_mnist = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=100, test=True, seed=42)
calibration_loader_mnist, _ = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=1, test=False, seed=42)

fp_model = TFC(64, 64, 64)
fp_model.load_state_dict(torch.load(ROOT_DIR+'/models/TFC.pt', map_location=torch.device('cpu')))

# force bias to zero
it = FullPrecisionModuleIterator(fp_model)
it.force_bias_zero()
fp_model.eval()

q_model = QTFCInt8(64, 64, 64)
prepare_brevitas_qmodel(fp_model, q_model, data_loader=calibration_loader_mnist, num_steps=300, seed=42)
print(q_model.get_quant_description((1, 1, 28, 28)))

# test model
test(q_model, test_loader_mnist, seed=0)

# export onnx
q_model.cpu()
export_finn_onnx(q_model, (1, 1, 28, 28), ROOT_DIR+'/models/FINN/QTFCInt8ZeroBias.onnx')
export_DOSA_onnx(q_model, (1, 1, 28, 28), ROOT_DIR+'/models/DOSA/QTFCInt8ZeroBias.onnx')


# check onnx model
model = onnx.load(ROOT_DIR+'/models/FINN/QTFCInt8ZeroBias.onnx')
onnx.checker.check_model(model)

# Export data used to test the model accuracy
export_data_as_npz(ROOT_DIR+'/data/mnist_test_data.npz', test_loader_mnist, num_batches=None,
                   feature_transform=lambda x: q_model.features[1](x).int(), dtype='int8', seed=0)
