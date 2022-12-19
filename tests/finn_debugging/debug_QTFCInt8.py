import torch
from torch import nn
import numpy as np
from src import data_loader, test
from src.definitions import ROOT_DIR
from src.models.full_precision import TFC
from src.models.quantized import QTFCInt8
from src.module_processing import FullPrecisionModuleIterator
from tests.torch_brevitas_comparisons.utils import prepare_brevitas_qmodel

# Prepare datasets
test_loader_mnist = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=100, test=True, seed=42)
calibration_loader_mnist, _ = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=1, test=False, seed=42)

# linear layers
lin_layers = [3, 8, 13, 18]

# bias zeroed model
fp_model = TFC(64, 64, 64)
fp_model.load_state_dict(torch.load(ROOT_DIR + '/models/TFC.pt', map_location=torch.device('cpu')))
it = FullPrecisionModuleIterator(fp_model)
it.force_bias_zero()
fp_model.eval()
q_model_bias_zero = QTFCInt8(64, 64, 64)
prepare_brevitas_qmodel(fp_model, q_model_bias_zero, data_loader=calibration_loader_mnist, num_steps=300, seed=42)
test(q_model_bias_zero, test_loader_mnist, seed=0)

# with bias model
fp_model.load_state_dict(torch.load(ROOT_DIR + '/models/TFC.pt', map_location=torch.device('cpu')))
q_model_with_bias = QTFCInt8(64, 64, 64)
prepare_brevitas_qmodel(fp_model, q_model_with_bias, data_loader=calibration_loader_mnist, num_steps=300, seed=42)
# remove bias of last layer
delattr(q_model_with_bias.features[18], 'bias_quant')
q_model_with_bias.features[18].bias = None
test(q_model_with_bias, test_loader_mnist, seed=0)

# with bias model, but bias removed afterwards
q_model = QTFCInt8(64, 64, 64)
prepare_brevitas_qmodel(fp_model, q_model, data_loader=calibration_loader_mnist, num_steps=300, seed=42)
for i in lin_layers:
    delattr(q_model.features[i], 'bias_quant')
    q_model.features[i].bias = None

print('\n --------------- Bias zeroed model ------------------')
print(q_model_bias_zero.get_quant_description((1, 1, 28, 28)))
print('\n ----------------------------------------------------\n')

print('\n --------------- With bias model ------------------')
print(q_model_with_bias.get_quant_description((1, 1, 28, 28)))
print('\n ----------------------------------------------------\n')

print('\n ----------- Bias removed afterwards model --------------')
print(q_model.get_quant_description((1, 1, 28, 28)))
print('\n')

print('\n --------------- Quant Biases ------------------')
for i in lin_layers:
    print(q_model_with_bias.features[i].int_bias(), '\n---')

print('\n --------------- Zero quant Biases ------------------')
for i in lin_layers:
    print(q_model_bias_zero.features[i].int_bias(), '\n---')


# ======= Inference =======
# make last layer return quant tensor
q_model_bias_zero.features[18].return_quant_tensor = True
q_model_with_bias.features[18].return_quant_tensor = True
q_model.features[18].return_quant_tensor = True

print('\n')
zero_input = torch.zeros((1, 1, 28, 28))
print('\n --- Zero Input ---')
print('zero bias model: ', q_model_bias_zero(zero_input).int())
print('model with bias: ', q_model_with_bias(zero_input).int())
print('model bias removed afterward: ', q_model(zero_input).int())


