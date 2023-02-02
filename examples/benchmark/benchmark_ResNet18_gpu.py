import os
import sys
import torch
import gpu as run
from gpu import calibrate_model
from gpu import TensorrtResNet18
from utils import parse_args, BenchmarkLogger


src_path = os.path.abspath('../../')
sys.path.insert(0, src_path)

import src
from src.definitions import ROOT_DIR
from src.models.full_precision.ResNet import ResNet18
from src import test


def prepare_test_data():
    return src.data_loader(data_dir=ROOT_DIR + '/data', dataset='CIFAR10', batch_size=100, test=True, seed=0)


def prepare_fp_model_and_train_data():
    # Prepare CIFAR10 dataset
    test_loader_cifar = src.data_loader(data_dir=ROOT_DIR + '/data', dataset='CIFAR10', batch_size=100, test=True,
                                        seed=0)

    # Prepare model
    model = ResNet18()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(ROOT_DIR + '/models/ResNet18.pt', map_location=device))
    model.eval()
    return model, test_loader_cifar


def prepare_q_model(fp_model, train_data):
    from pytorch_quantization import quant_modules
    quant_modules.initialize()
    q_model = TensorrtResNet18()

    # rename state dict for compatibility
    q_model.load_state_dict(fp_model.state_dict())

    # TODO For unknown reason, calibration makes accuracy drop significantly
    # calibrate_model(q_model, train_data)
    return q_model


def main():
    log_file, sleep_interval, run_interval = parse_args('ResNet18', 'gpu')
    logger = BenchmarkLogger('ResNet18', log_file)

    # first prepare full precision model
    fp_model, train_data = prepare_fp_model_and_train_data()
    q_model = prepare_q_model(fp_model, train_data)

    test_data = prepare_test_data()
    test(fp_model, test_data, seed=0, verbose=True)
    test(q_model, test_data, seed=0, verbose=True)

    # prepare model engines for each batch size, full-precision and int8 precision
    batch_sizes = [1, 10, 100, 1000]
    fp_model_description = 'full precision model'
    q_model_description = 'int8 quantized model'
    models = run.prepare_engines_and_contexts('ResNet18', fp_model, q_model, batch_sizes, (3, 32, 32), train_data,
                                              fp_model_description, q_model_description)
    # Accuracy
    logger.write_section_accuracy()
    test_data = prepare_test_data()
    run.compute_models_accuracy(models, test_data, logger)

    # Runtimes
    logger.write_section_runtime()
    run.compute_models_runtime(models, (1, 28, 28), batch_sizes, logger)

    # Empty run
    # Model do inference for a few minutes with each batch size, you are supposed to use a system monitoring tool
    # in parallel to collect hardware data
    logger.write_section_empty_run(sleep_interval, run_interval)
    run.empty_run_models(models, batch_sizes, logger, sleep_interval, run_interval)

    # write to file
    logger.close()


if __name__ == "__main__":
    main()
