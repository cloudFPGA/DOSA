import torch
import tensorrt as trt
import pytorch_quantization.nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

quant_nn.TensorQuantizer.use_fb_fake_quant = True
TRT_LOGGER = trt.Logger()

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)


def collect_stats(model, dataloader, device, seed=0):
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # calibrate
    count = 0
    torch.manual_seed(seed)
    for features, _ in dataloader:
        if count >= 10:
            break
        model(features.to(device))
        count += 1

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, device, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.to(device)


def calibrate_model(model, dataloader, seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        collect_stats(model, dataloader, device, seed)
        compute_amax(model, device, method='percentile', percentile=99.99)


def build_engine(onnx_file_path, use_int8=True):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch_flag) as network, \
            builder.create_builder_config() as builder_config:

        if use_int8:
            assert builder.platform_has_fast_int8, 'ERROR: platform do not support int8'
            builder_config.max_workspace_size = 1 << 30
            builder.max_batch_size = 1
            builder_config.set_flag(trt.BuilderFlag.INT8)

        with open(onnx_file_path, 'rb') as model:
            parser = trt.OnnxParser(network, TRT_LOGGER)
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        engine = builder.build_engine(network, builder_config)
        context = engine.create_execution_context()

        return engine, context
