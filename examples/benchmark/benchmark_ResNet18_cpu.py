import torch
import torch.quantization
from torch import nn
from utils import parse_args, BenchmarkLogger
import cpu as run

import src
from src.definitions import ROOT_DIR
from src.models.full_precision.ResNet import ResidualBlock, ResNet
from cpu import prepare_int8_dynamic_qmodel, prepare_int8_static_qmodel


class TorchQuantResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, stride=1):
        super(TorchQuantResidualBlock, self).__init__(in_channels, out_channels, stride)
        self.skip_add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.skip_add_relu.add_relu(out, self.downsample(x))
        return out


def prepare_test_data():
    return src.data_loader(data_dir=ROOT_DIR + '/data', dataset='CIFAR10', batch_size=100, test=True, seed=0)


def prepare_fp_model_and_train_data():
    # Prepare CIFAR10 dataset
    train_loader, _ = src.data_loader(data_dir=ROOT_DIR + '/data', dataset='CIFAR10', batch_size=100, test=False,
                                      seed=0)

    # Prepare model
    model = ResNet(TorchQuantResidualBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(ROOT_DIR + '/models/ResNet18.pt', map_location='cpu'))
    model.eval()
    return model, train_loader


def prepare_static_qmodel(fp_model, train_data):
    modules_to_fuse = [['conv1.0', 'conv1.1', 'conv1.2'],
                       # layer 0
                       ['layer0.0.conv1.0', 'layer0.0.conv1.1', 'layer0.0.conv1.2'],
                       ['layer0.0.conv2.0', 'layer0.0.conv2.1'],
                       ['layer0.1.conv1.0', 'layer0.1.conv1.1', 'layer0.1.conv1.2'],
                       ['layer0.1.conv2.0', 'layer0.1.conv2.1'],
                       # layer 1
                       ['layer1.0.conv1.0', 'layer1.0.conv1.1', 'layer1.0.conv1.2'],
                       ['layer1.0.conv2.0', 'layer1.0.conv2.1'],
                       ['layer1.0.downsample.0', 'layer1.0.downsample.1'],
                       ['layer1.1.conv1.0', 'layer1.1.conv1.1', 'layer1.1.conv1.2'],
                       ['layer1.1.conv2.0', 'layer1.1.conv2.1'],
                       # layer 2
                       ['layer2.0.conv1.0', 'layer2.0.conv1.1', 'layer2.0.conv1.2'],
                       ['layer2.0.conv2.0', 'layer2.0.conv2.1'],
                       ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
                       ['layer2.1.conv1.0', 'layer2.1.conv1.1', 'layer2.1.conv1.2'],
                       ['layer2.1.conv2.0', 'layer2.1.conv2.1'],
                       # layer 3
                       ['layer3.0.conv1.0', 'layer3.0.conv1.1', 'layer3.0.conv1.2'],
                       ['layer3.0.conv2.0', 'layer3.0.conv2.1'],
                       ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
                       ['layer3.1.conv1.0', 'layer3.1.conv1.1', 'layer3.1.conv1.2'],
                       ['layer3.1.conv2.0', 'layer3.1.conv2.1']]
    return prepare_int8_static_qmodel(fp_model, train_data, modules_to_fuse)


def prepare_dynamic_qmodel(fp_model):
    # "Dynamic quantization is currently supported only for nn.Linear and nn.LSTM"
    # https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic
    # return prepare_int8_dynamic_qmodel(fp_model, {nn.Linear, nn.Conv2d})
    return prepare_int8_dynamic_qmodel(fp_model, {nn.Linear})


def main():
    log_file, sleep_interval, run_interval = parse_args('ResNet18', 'cpu')
    logger = BenchmarkLogger('ResNet18', log_file)

    # prepare models
    fp_model, train_data = prepare_fp_model_and_train_data()
    q_dyn_model = prepare_dynamic_qmodel(fp_model)
    q_static_model = prepare_static_qmodel(fp_model, train_data)
    models = {
        'full precision model': fp_model,
        'dynamic int8 model': q_dyn_model,
        'static int8 model': q_static_model
    }

    # batch sizes to test
    batch_sizes = [1, 10, 100, 1000]

    # Size
    logger.write_section_size()
    run.compute_models_size(models, logger)

    # Accuracy
    logger.write_section_accuracy()
    test_data = prepare_test_data()
    run.compute_models_accuracy(models, test_data, logger)

    # Runtimes
    logger.write_section_runtime()
    run.compute_models_runtime(models, (3, 32, 32), batch_sizes, logger)

    # Empty run
    # Model do inference for a few minutes with each batch size, you are supposed to use a system monitoring tool
    # in parallel to collect hardware data
    run.empty_run_models(models, (3, 32, 32), batch_sizes, logger, sleep_interval, run_interval)

    # write to file
    logger.close()


if __name__ == "__main__":
    main()
