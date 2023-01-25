import copy
import os
import sys
import torch
import argparse
import torch.quantization

src_path = os.path.abspath('../../../')
sys.path.insert(0, src_path)

import src
from src import calibrate
from src.definitions import ROOT_DIR
from src.models.full_precision import TFC
from src.utils.monitoring import write_model_stats, run_model


def prepare_fp_model_and_dataloader():
    # Prepare MNIST dataset
    test_loader_mnist = src.data_loader(data_dir=ROOT_DIR + '/data', dataset='MNIST', batch_size=100, test=True, seed=0)

    # Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TFC(64, 64, 64)
    model.load_state_dict(torch.load(ROOT_DIR + '/models/TFC.pt', map_location=device))
    model.eval()
    return model, test_loader_mnist


def prepare_int8_static_qmodel(fp_model, dataloader):
    from torch.quantization.qconfig import QConfig
    from torch.quantization.observer import MinMaxObserver
    from torch_quant import StaticQuantModel

    internal_model = copy.deepcopy(fp_model)
    internal_model = torch.quantization.fuse_modules(
        internal_model,
        [['1', '2'], ['5', '6'], ['9', '10']],
        inplace=True
    )

    my_qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=True)
    )

    q_model = StaticQuantModel(internal_model)
    q_model.qconfig = my_qconfig
    torch.ao.quantization.prepare(q_model, inplace=True)
    calibrate(q_model, dataloader, num_steps=1, seed=42)
    torch.ao.quantization.convert(q_model, inplace=True)
    return q_model


def prepare_int8_dynamic_qmodel(fp_model):
    import torch.nn as nn
    return torch.quantization.quantize_dynamic(
        fp_model, {nn.Linear}, dtype=torch.qint8
    )


def main():
    parser = argparse.ArgumentParser(
        description="Monitor TFC models. This script is supposed to be launched in parallel to a gpu or cpu system "
                    "management tool, to measure hardware power consumption while the script is running (eg. "
                    "nvidia-smi)."
    )
    parser.add_argument(
        "--log_file", help="file path of the running logs.", type=str, required=True
    )
    parser.add_argument(
        "--sleep_interval", help="sleep interval in seconds before two inference sessions", type=int, default=300
    )
    parser.add_argument(
        "--running_interval", help="target running time in seconds", type=int, default=300
    )

    # parse arguments
    args = parser.parse_args()
    log_file = args.log_file
    sleep_interval = args.sleep_interval
    running_interval = args.running_interval

    if os.path.isfile(log_file):
        print('{} already exists, deleting it...'.format(log_file))
        os.remove(log_file)
    f = open(log_file, 'w')

    fp_model, dataloader = prepare_fp_model_and_dataloader()
    q_dyn_model = prepare_int8_dynamic_qmodel(fp_model)
    q_static_model = prepare_int8_static_qmodel(fp_model, dataloader)

    # (size, accuracy) of models
    write_model_stats(fp_model, dataloader, f, 'full precision model')
    write_model_stats(q_dyn_model, dataloader, f, 'dynamic int8 model')
    write_model_stats(q_static_model, dataloader, f, 'static int8 model')

    # running the models
    f.write('\n')
    run_model(fp_model, dataloader, sleep_interval, running_interval, f, 'full precision model')
    run_model(q_dyn_model, dataloader, sleep_interval, running_interval, f, 'dynamic int8 model')
    run_model(q_static_model, dataloader, sleep_interval, running_interval, f, 'static int8 model')

    f.close()


if __name__ == "__main__":
    main()
