import torch

from src import data_loader, test
from src.module_processing import FullPrecisionModuleIterator
from src.models.full_precision import ResNet18
from src.models import quantized
from tests.torch_brevitas_comparisons.utils import prepare_brevitas_qmodel

# Prepare datasets
torch.manual_seed(0)
test_loader_cifar = data_loader(data_dir='../data', dataset='CIFAR10', batch_size=128, test=True)

torch.manual_seed(0)
calibration_loader_cifar, _ = data_loader(data_dir='../data', dataset='CIFAR10', batch_size=1, test=False)

fp_model = ResNet18()
fp_model.load_state_dict(torch.load('../models/ResNet18.pt', map_location=torch.device('cpu')))


# ======================= Uncomment one of below lines ======================
# Full precision: 91.49%
brevitas_quant_model = quantized.QResNet18()  # ( %)
# ===========================================================================

# load and calibrate quantized model
torch.manual_seed(0)
prepare_brevitas_qmodel(fp_model, brevitas_quant_model, data_loader=calibration_loader_cifar, num_steps=300)

print('\n ----------------------------------------------------\n')
print(brevitas_quant_model.get_quant_description())
print('\n ----------------------------------------------------\n')

# accuracies
print('--- Full Precision accuracy ---')
test(fp_model, test_loader_cifar)

print('\n--- Quantized model accuracy ---')
# test(brevitas_quant_model, test_loader_mnist)

# brevitas_quant_model.collect_stats(test_loader_cifar, 10)

