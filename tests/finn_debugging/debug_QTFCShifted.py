import torch

from src import data_loader, test
from src.definitions import ROOT_DIR
from src.models.full_precision import TFC
from src.models import quantized
from src.module_processing import FullPrecisionModuleIterator
from tests.torch_brevitas_comparisons.utils import prepare_brevitas_qmodel

# Prepare datasets
test_loader_mnist = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=100, test=True, seed=42)
calibration_loader_mnist, _ = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=1, test=False, seed=42)

fp_model = TFC(64, 64, 64)
fp_model.load_state_dict(torch.load(ROOT_DIR+'/models/TFC.pt', map_location=torch.device('cpu')))
it = FullPrecisionModuleIterator(fp_model)
it.force_bias_zero()
fp_model.eval()

# q_model = quantized.QTFCShiftedQuantAct8(64, 64, 64)  # (98.14%, 98.14%)
q_model = quantized.QTFCShiftedQuantAct4(64, 64, 64)  # (97.55 %, 97.55%)

# load and calibrate quantized model
prepare_brevitas_qmodel(fp_model, q_model, data_loader=calibration_loader_mnist, num_steps=300, seed=42)

print('\n ----------------------------------------------------\n')
print(q_model.get_quant_description((1, 1, 28, 28)))
print('\n ----------------------------------------------------\n')

input_data = torch.full((1, 28, 28), 0.1)
print(q_model(input_data))

