import torch

from src import data_loader, test
from src.module_processing import FullPrecisionModuleIterator
from src.models.full_precision import TFC
from src.models import quantized
from tests.torch_brevitas_comparisons.utils import prepare_brevitas_qmodel

# Prepare datasets
torch.manual_seed(0)
test_loader_mnist = data_loader(data_dir='../data', dataset='MNIST', batch_size=100, test=True)

torch.manual_seed(0)
calibration_loader_mnist, _ = data_loader(data_dir='../data', dataset='MNIST', batch_size=1, test=False)

fp_model = TFC(64, 64, 64)
fp_model.load_state_dict(torch.load('../models/TFC.pt', map_location=torch.device('cpu')))


# ------------------ Uncomment to force model bias to zero ------------------
# it = FullPrecisionModuleIterator(fp_model)
# it.force_bias_zero()
# fp_model.eval()
# ---------------------------------------------------------------------------

# ======================= Uncomment one of below lines ======================
# Full precision: (bias zeroed: 98.14%, bias: 98.12%)
# brevitas_quant_model = quantized.QTFC(64, 64, 64)  # (98.14 %, 98.12%)
# brevitas_quant_model = quantized.QTFCInt8(64, 64, 64)  # (98.09 %, 98.10%)
# brevitas_quant_model = quantized.QTFCInt5(64, 64, 64)  # (98.00%, 98.00%)
# brevitas_quant_model = quantized.QTFCInt4(64, 64, 64)  # (97.37%, 97.38%)
# brevitas_quant_model = quantized.QTFCInt3(64, 64, 64)  # (96.4%, 96.42%)
# brevitas_quant_model = quantized.QTFCFixedPoint8(64, 64, 64)  # (98.08%, 98.09%)
# brevitas_quant_model = quantized.QTFCFixedPoint5(64, 64, 64)  # (97.72%, 97.74%)
brevitas_quant_model = quantized.QTFCFixedPoint4(64, 64, 64)  # (94.36%, 94.41%)
# brevitas_quant_model = quantized.QTFCFixedPoint3(64, 64, 64)  # (21.05%, 21.78%)
# brevitas_quant_model = quantized.QTFCShiftedQuantAct8(64, 64, 64)  # (98.14%, 98.14%)
# brevitas_quant_model = quantized.QTFCShiftedQuantAct4(64, 64, 64)  # (97.56 %, 97.55%)
# brevitas_quant_model = quantized.QTFCTernary(64, 64, 64)  # (9.8 %)
# brevitas_quant_model = quantized.QTFCBinary(64, 64, 64)  # (9.82 %)
# ===========================================================================

# load and calibrate quantized model
torch.manual_seed(0)
prepare_brevitas_qmodel(fp_model, brevitas_quant_model, data_loader=calibration_loader_mnist, num_steps=300)

print('\n ----------------------------------------------------\n')
print(brevitas_quant_model.get_quant_description((1, 1, 28, 28)))
print('\n ----------------------------------------------------\n')

# accuracies
print('--- Full Precision accuracy ---')
test(fp_model, test_loader_mnist)

print('\n--- Quantized model accuracy ---')
test(brevitas_quant_model, test_loader_mnist)

# Collect statistics
print('\nCollecting statistics...')
brevitas_quant_model.collect_stats(test_loader_mnist, 10)

