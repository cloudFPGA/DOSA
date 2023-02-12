import argparse
import sys

import torch
import onnx

from dnn_quant.data import export_data_as_npz
from dnn_quant.definitions import ROOT_DIR
from dnn_quant import data_loader, test
from dnn_quant.module_processing import FullPrecisionModuleIterator
from dnn_quant.models.full_precision.TFC import TFC
from dnn_quant.models import quantized
from dnn_quant.onnx import export_DOSA_onnx, export_FINN_onnx


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate QTFC model and export it to FINN or DOSA along with quantized input data.'
    )
    parser.add_argument(
        '--model', help='model to use (eg. QTFCInt4)', type=str, required=True
    )
    parser.add_argument(
        '--with-bias', help='keep the bias in the model', dest='bias', action='store_true', default=True
    )
    parser.add_argument(
        '--zero-bias', help='set the bias in the model to zero', dest='bias', action='store_false'
    )
    parser.add_argument(
        '--compiler', help='the compiler to export to (FINN or DOSA)', choices=['FINN', 'DOSA'], default='FINN'
    )
    parser.add_argument(
        '--export-input', help='export the quantized input data (available for FINN only)', dest='export_input',
        action='store_true', default=False
    )
    parser.add_argument(
        '--no-export-input', help='not exporting the quantized input data', dest='export_input', action='store_false'
    )
    parser.add_argument(
        "--verbose", help="display computed per-layer quantization parameters", action='store_true', default=False
    )

    # parse arguments
    args = parser.parse_args()
    model_name = args.model
    bias = args.bias
    compiler = args.compiler
    export_input = args.export_input
    verbose = args.verbose

    assert model_name.find('QTFC') == 0, f"ERROR: model name needs to start with 'QTFC'"

    try:
        model_class = getattr(quantized, model_name)
        q_model = model_class(64, 64, 64)

    except AttributeError:
        print(f"ERROR: {model_name} does not exist.", file=sys.stderr)
        sys.exit(1)

    if compiler == 'DOSA':
        print(f"ERROR: streamlining step of DOSA export does not work for QResNet yet. Interrupt.")
        sys.exit(1)

    return model_name, q_model, bias, compiler, export_input, verbose


def get_QTFC_FINN_input_mapping_function(q_model):
    def mapping_function(x):
        x = q_model.features[0](x)  # reshape
        x = q_model.features[1](x).int()  # quantize data: fp32 -> int8
        return x

    return mapping_function


def main():
    model_name, q_model, bias, compiler, export_input, verbose = parse_args()

    # Prepare MNIST dataset
    test_loader_mnist = data_loader(data_dir=f"{ROOT_DIR}/data", dataset='MNIST', batch_size=100, test=True, seed=42)
    calibration_loader_mnist, _ = data_loader(data_dir=ROOT_DIR+'/data', dataset='MNIST', batch_size=1, test=False,
                                              seed=42)

    # full precision model
    fp_model = TFC(64, 64, 64)
    fp_model.load_state_dict(torch.load(f"{ROOT_DIR}/models/TFC.pt", map_location=torch.device('cpu')))

    if not bias:
        # set bias to zero
        it = FullPrecisionModuleIterator(fp_model)
        it.force_bias_zero()
        fp_model.eval()

    print("Preparing the model...")
    q_model.load_state_and_calibrate(fp_model, data_loader=calibration_loader_mnist, num_steps=300, seed=42)
    print(f"Model {model_name}{'' if bias else ' (with bias set to zero)'} ready.")

    if verbose:
        print(f"\n----- {model_name} Description -----")
        print(q_model.get_quant_description((1, 1, 28, 28)))

    # test model
    accuracy = test(q_model, test_loader_mnist, seed=0, verbose=False)
    num_img = len(test_loader_mnist) * test_loader_mnist.batch_size
    print(f"\nAccuracy of {model_name} with {'' if not bias else 'zero'} bias on {num_img} test images: {accuracy}%.\n")

    # export onnx
    q_model.cpu()
    onnx_model_path = f"{ROOT_DIR}/models/{compiler}/{model_name}{'WithBias' if bias else 'ZeroBias'}.onnx"

    if compiler == 'FINN':
        export_FINN_onnx(module=q_model, input_shape=(1, 1, 28, 28), export_path=onnx_model_path)
    else:
        export_DOSA_onnx(module=q_model, input_shape=(1, 1, 28, 28), export_path=onnx_model_path)

    # check onnx model
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    print(f"Model successfully exported to : {onnx_model_path}")

    # Export data used to test the model accuracy
    if export_input:
        feature_transform_function = get_QTFC_FINN_input_mapping_function(q_model)
        data_path = f"{ROOT_DIR}/data/mnist_test_data.npz"
        dtype = 'int8' if not isinstance(q_model, quantized.QTFCShiftedQuantAct8) else 'int32'
        export_data_as_npz(data_path, test_loader_mnist, num_batches=None, feature_transform=feature_transform_function,
                           dtype=dtype, seed=0)
        print(f"Quantized input data for FINN driver exported to {data_path}")


if __name__ == "__main__":
    main()
