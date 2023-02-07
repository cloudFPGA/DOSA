import torch

from dnn_quant import data_loader, test
from dnn_quant.definitions import ROOT_DIR
from dnn_quant.module_processing import FullPrecisionModuleIterator
from dnn_quant.models.full_precision import TFC
from dnn_quant.models import quantized

# Prepare datasets
test_loader_mnist = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=100, test=True, seed=42)
calibration_loader_mnist, _ = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=1, test=False, seed=42)

fp_model = TFC(64, 64, 64)
fp_model.load_state_dict(torch.load(ROOT_DIR+'/models/TFC.pt', map_location=torch.device('cpu')))


# ------------------ Uncomment to force model bias to zero ------------------
it = FullPrecisionModuleIterator(fp_model)
it.force_bias_zero()
fp_model.eval()
# ---------------------------------------------------------------------------

# ======================= Uncomment one of below lines ======================
# Full precision: (bias zeroed: 98.14%, with bias: 98.12%)
# q_model = quantized.QTFC(64, 64, 64)  # (98.14 %, 98.12%)
q_model = quantized.QTFCInt8(64, 64, 64)  # (98.09 %, 98.10%)
# q_model = quantized.QTFCInt5(64, 64, 64)  # (97.99%, 98.00%)
# q_model = quantized.QTFCInt4(64, 64, 64)  # (97.37%, 97.37%)
# q_model = quantized.QTFCInt3(64, 64, 64)  # (96.41%, 96.42%)
# q_model = quantized.QTFCFixedPoint8(64, 64, 64)  # (98.08%, 98.09%)
# q_model = quantized.QTFCFixedPoint5(64, 64, 64)  # (97.72%, 97.74%)
# q_model = quantized.QTFCFixedPoint4(64, 64, 64)  # (94.35%, 94.40%)
# q_model = quantized.QTFCFixedPoint3(64, 64, 64)  # (21.09%, 21.75%)
# q_model = quantized.QTFCShiftedQuantAct8(64, 64, 64)  # (98.14%, 98.14%)
# q_model = quantized.QTFCShiftedQuantAct4(64, 64, 64)  # (97.55 %, 97.55%)
# q_model = quantized.QTFCTernary(64, 64, 64)  # (9.80 %, 9.80 %)
# q_model = quantized.QTFCBinary(64, 64, 64)  # (9.82%, 9.82 %)
# q_model = quantized.QTFCMixed(64, 64, 64)  # (98.02%, 98.02%)
# ===========================================================================

# load and calibrate quantized model
q_model.load_state_and_calibrate(fp_model, data_loader=calibration_loader_mnist, num_steps=300, seed=42)

print('\n ----------------------------------------------------\n')
print(q_model.get_quant_description((1, 1, 28, 28)))
print('\n ----------------------------------------------------\n')

seed = 0
# accuracies
print('--- Full Precision accuracy ---')
test(fp_model, test_loader_mnist, seed=seed)

print('\n--- Quantized model accuracy ---')
test(q_model, test_loader_mnist, seed=seed)

# Collect statistics
print('\nCollecting statistics...')
q_model.collect_stats(test_loader_mnist, 10, seed=seed)


