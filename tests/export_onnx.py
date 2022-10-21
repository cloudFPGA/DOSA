import torch

import os
import sys

src_path = os.path.abspath('../')
sys.path.insert(0, src_path)

from src.data import data_loader
from src.test import test

from src.models.full_precision.TFC import TFC
from src.models.quantized.QTFC.QTFC_scaled_int import QTFC

# Prepare MNIST dataset
test_loader_mnist = data_loader(data_dir='../data', dataset='MNIST', batch_size=100, test=True)

model = TFC(64, 64, 64)
model.load_state_dict(torch.load('../models/TFC.pt', map_location=torch.device('cpu')))

from brevitas import config

config.IGNORE_MISSING_KEYS = True
brevitas_qmodel = QTFC(64, 64, 64)
brevitas_qmodel.load_model_state_dict(model)

test(brevitas_qmodel, test_loader_mnist)

from brevitas.export import StdQOpONNXManager

StdQOpONNXManager.export(brevitas_qmodel,
                         input_shape=(100, 1, 28, 28),
                         export_path='../models/{}.onnx'.format('TFF.quant_brevitas.onnx'))
