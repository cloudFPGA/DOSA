import torch

from src import data_loader, test
from src.definitions import ROOT_DIR
from src.module_processing import FullPrecisionModuleIterator
from src.models.full_precision import ResNet18
from src.models import quantized
from tests.torch_brevitas_comparisons.utils import prepare_brevitas_qmodel

# Prepare datasets
test_loader_cifar = data_loader(data_dir=ROOT_DIR+'/data', dataset='CIFAR10', batch_size=128, test=True, seed=0)
calibration_loader_cifar, _ = data_loader(data_dir=ROOT_DIR+'/data', dataset='CIFAR10',
                                          batch_size=1, test=False, seed=0)

fp_model = ResNet18()
fp_model.load_state_dict(torch.load(ROOT_DIR+'/models/ResNet18.pt', map_location=torch.device('cpu')))


# ======================= Uncomment one of below lines =======================
# Full precision: 91.47%
# brevitas_quant_model = quantized.QResNet18()  # (91.47%)
# brevitas_quant_model = quantized.QResNet18Int8()  # (90.95%)
# brevitas_quant_model = quantized.QResNet18Int5()  # (86.38%)
brevitas_quant_model = quantized.QResNet18Int4()  # (50.86%)
# ============================================================================

# load and calibrate quantized model
prepare_brevitas_qmodel(fp_model, brevitas_quant_model, data_loader=calibration_loader_cifar, num_steps=300, seed=0)

print('\n ----------------------------------------------------\n')
print(brevitas_quant_model.get_quant_description((1, 3, 32, 32)))
print('\n ----------------------------------------------------\n')

# accuracies
print('--- Full Precision accuracy ---')
seed = 0
# test(fp_model, test_loader_cifar, seed=seed)

print('\n--- Quantized model accuracy ---')
test(brevitas_quant_model, test_loader_cifar, seed=seed)

# Collect statistics
print('\nCollecting statistics...')
brevitas_quant_model.collect_stats(test_loader_cifar, 10, seed=seed)

