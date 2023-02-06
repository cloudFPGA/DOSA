import torch
import numpy as np
import onnx
from brevitas.export import export_finn_onnx

from dnn_quant.data import export_data_as_npz
from dnn_quant.definitions import ROOT_DIR
from dnn_quant import data_loader, test
from dnn_quant.module_processing import FullPrecisionModuleIterator
from dnn_quant.models.full_precision.TFC import TFC
from dnn_quant.models.quantized import QTFCInt8
from dnn_quant.onnx import export_DOSA_onnx


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
q_model.load_state_and_calibrate(fp_model, data_loader=calibration_loader_mnist, num_steps=300, seed=42)
print(q_model.get_quant_description((1, 1, 28, 28)))

# test model
test(q_model, test_loader_mnist, seed=0)

# export onnx
q_model.cpu()
b = export_finn_onnx(module=q_model, input_shape=(1, 1, 28, 28), export_path=ROOT_DIR+'/models/FINN/QTFCInt8ZeroBias.onnx')
export_DOSA_onnx(module=q_model, input_shape=(1, 1, 28, 28), export_path=ROOT_DIR+'/models/DOSA/QTFCInt8ZeroBias.onnx')

# check onnx model
model = onnx.load(ROOT_DIR+'/models/FINN/QTFCInt8ZeroBias.onnx')
model = onnx.load(ROOT_DIR+'/models/DOSA/QTFCInt8ZeroBias.onnx')
onnx.checker.check_model(model)


# Export data used to test the model accuracy
export_data_as_npz(ROOT_DIR+'/data/mnist_test_data.npz', test_loader_mnist, num_batches=None,
                   feature_transform=lambda x: q_model.features[1](x).int(), dtype='int8', seed=0)
