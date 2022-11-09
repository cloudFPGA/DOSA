import torch

from src import data_loader, test
from src.model_processing import FullPrecisionModelIterator
from src.models.full_precision.TFC import TFC
from src.models.quantized import QTFCInt8
import brevitas.onnx as bo


# Prepare MNIST dataset
torch.manual_seed(0)
test_loader_mnist = data_loader(data_dir='../data', dataset='MNIST', batch_size=100, test=True)

fp_model = TFC(64, 64, 64)
fp_model.load_state_dict(torch.load('../models/TFC.pt', map_location=torch.device('cpu')))

# force bias to zero
it = FullPrecisionModelIterator(fp_model)
it.force_bias_zero()
fp_model.eval()

q_model = QTFCInt8(64, 64, 64)
q_model.load_model_state_dict(fp_model)

# test model
test(q_model, test_loader_mnist)

# export onnx
q_model.cpu()
bo.export_finn_onnx(q_model, (1, 1, 28, 28), '../models/QTFCInt8.onnx')


