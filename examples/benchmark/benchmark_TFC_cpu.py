import torch
import torch.quantization
from torch import nn
from utils import parse_args, BenchmarkLogger
import cpu as run

import src
from src.definitions import ROOT_DIR
from src.models.full_precision import TFC
from cpu import prepare_int8_dynamic_qmodel, prepare_int8_static_qmodel


def prepare_test_data():
    return src.data_loader(data_dir=ROOT_DIR + '/data', dataset='MNIST', batch_size=100, test=True, seed=0)


def prepare_fp_model_and_train_data():
    # Prepare MNIST dataset
    train_loader, _ = src.data_loader(data_dir=ROOT_DIR + '/data', dataset='MNIST', batch_size=100, test=False, seed=0)

    # Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TFC(64, 64, 64)
    model.load_state_dict(torch.load(ROOT_DIR + '/models/TFC.pt', map_location=device))
    model.eval()
    return model, train_loader


def prepare_static_qmodel(fp_model, train_data):
    modules_to_fuse = [['1', '2'], ['5', '6'], ['9', '10']]
    return prepare_int8_static_qmodel(fp_model, train_data, modules_to_fuse)


def prepare_dynamic_qmodel(fp_model):
    return prepare_int8_dynamic_qmodel(fp_model, {nn.Linear})


def main():
    log_file, sleep_interval, run_interval = parse_args('TFC', 'cpu')
    logger = BenchmarkLogger('TFC', log_file)

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
    batch_sizes = [1, 100, 1000, 10000]

    # Size
    logger.write_section_size()
    run.compute_models_size(models, logger)

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
    run.empty_run_models(models, (1, 28, 28), batch_sizes, logger, sleep_interval, run_interval)

    # write to file
    logger.close()


if __name__ == "__main__":
    main()
