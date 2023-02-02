import os
import sys
import torch
from torch import nn
import pytorch_quantization.nn as quant_nn
import gpu as run
from gpu import calibrate_model
from utils import parse_args, BenchmarkLogger


src_path = os.path.abspath('../../')
sys.path.insert(0, src_path)

import src
from src.definitions import ROOT_DIR
from src.models.full_precision import TFC


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


def prepare_test_data():
    return src.data_loader(data_dir=ROOT_DIR + '/data', dataset='MNIST', batch_size=100, test=True, seed=0)


def prepare_fp_model_and_train_data():
    # Prepare MNIST dataset
    train_data, _ = src.data_loader(data_dir=ROOT_DIR + '/data', dataset='MNIST', batch_size=100, test=False, seed=0)

    # Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TFC(64, 64, 64)
    model.load_state_dict(torch.load(ROOT_DIR + '/models/TFC.pt', map_location=device))
    model.eval()
    return model, train_data


def prepare_q_model(fp_model, train_data):
    from pytorch_quantization import quant_modules
    quant_modules.initialize()
    q_model = TensorrtTFC(64, 64, 64)

    # rename state dict for compatibility
    q_state_dict = q_model.state_dict()
    fp_state_dict = fp_model.state_dict()
    renamed_state_dict = {a: fp_state_dict[b] for a, b in zip(q_state_dict, fp_state_dict)}

    q_model.load_state_dict(renamed_state_dict)
    calibrate_model(q_model, train_data)
    return q_model


def main():
    log_file, sleep_interval, run_interval = parse_args('TFC', 'gpu')
    logger = BenchmarkLogger('TFC', log_file)

    # prepare full precision and quantized model
    fp_model, train_data = prepare_fp_model_and_train_data()
    q_model = prepare_q_model(fp_model, train_data)

    # prepare model engines for each batch size, full-precision and int8 precision
    batch_sizes = [1, 100, 1000, 10000]
    fp_model_description = 'full precision model'
    q_model_description = 'int8 quantized model'
    models = run.prepare_engines_and_contexts('TFC', fp_model, q_model, batch_sizes, (1, 28, 28),
                                              fp_model_description, q_model_description)

    # Accuracy
    logger.write_section_accuracy()
    test_data = prepare_test_data()
    run.compute_models_accuracy(models, test_data, logger)

    # Runtimes
    logger.write_section_runtime()
    run.compute_models_runtime(models, batch_sizes, logger)

    # Empty run
    # Model do inference for a few minutes with each batch size, you are supposed to use a system monitoring tool
    # in parallel to collect hardware data
    logger.write_section_empty_run(sleep_interval, run_interval)
    run.empty_run_models(models, batch_sizes, logger, sleep_interval, run_interval)

    # write to file
    logger.close()


if __name__ == "__main__":
    main()
