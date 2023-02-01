import argparse
import os
import sys
import torch
from torch import nn
import pytorch_quantization.nn as quant_nn
from tensorrt_quant import calibrate_model, export_onnx, build_engine
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


src_path = os.path.abspath('../../')
sys.path.insert(0, src_path)

import src
from src.definitions import ROOT_DIR
from src.models.full_precision import TFC
from src import test


dropout = 0.2
in_features = 28 * 28


class TensorrtTFC(nn.Sequential):

    def __init__(self, hidden1, hidden2, hidden3):
        super(TensorrtTFC, self).__init__(
            quant_nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            quant_nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            quant_nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            quant_nn.Linear(hidden3, 10)
        )

    def forward(self, x):
        x = x.reshape((-1, in_features))
        return super(TensorrtTFC, self).forward(x)


def rename_state_dict(to_rename_state_dict, target_state_dict):
    new_dict = {a: to_rename_state_dict[b] for a, b in zip(target_state_dict, to_rename_state_dict)}
    return new_dict


def prepare_fp_model_and_dataloader():
    # Prepare MNIST dataset
    test_loader_mnist = src.data_loader(data_dir=ROOT_DIR + '/data', dataset='MNIST', batch_size=100, test=True, seed=0)

    # Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TFC(64, 64, 64)
    model.load_state_dict(torch.load(ROOT_DIR + '/models/TFC.pt', map_location=device))
    model.eval()
    return model, test_loader_mnist


def prepare_q_model(fp_model, dataloader):
    from pytorch_quantization import quant_modules
    quant_modules.initialize()
    q_model = TensorrtTFC(64, 64, 64)

    # rename state dict for compatibility
    q_state_dict = q_model.state_dict()
    fp_state_dict = fp_model.state_dict()
    renamed_state_dict = {a: fp_state_dict[b] for a, b in zip(q_state_dict, fp_state_dict)}

    q_model.load_state_dict(renamed_state_dict)
    calibrate_model(q_model, dataloader)
    return q_model


def main():
    fp_model, dataloader = prepare_fp_model_and_dataloader()
    q_model = prepare_q_model(fp_model, dataloader)

    test(fp_model, dataloader, seed=0, verbose=True)
    test(q_model, dataloader, seed=0, verbose=True)

    onnx_file_path = 'quant_tfc.onnx'
    export_onnx(q_model, onnx_file_path, (1, 1, 28, 28))
    engine, context = build_engine(onnx_file_path)

    context.set_binding_shape(engine.get_binding_index("input"), (1, 1, 28, 28))
    # Allocate host and device buffers
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            image, label = next(iter(dataloader))
            image = image[[0]]
            label = label[[0]]
            input_buffer = np.ascontiguousarray(image)
            input_memory = cuda.mem_alloc(image.nbytes)
            bindings.append(int(input_memory))
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))

    stream = cuda.Stream()
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(input_memory, input_buffer, stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer prediction output from the GPU.
    cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
    # Synchronize the stream
    stream.synchronize()
    print(output_buffer)
    print(label)


if __name__ == "__main__":
    main()
