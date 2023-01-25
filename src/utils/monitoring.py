import os
import torch
import time

from src import test


def get_model_size(model):
    torch.save(model.state_dict(), 'temp.p')
    size = os.path.getsize('temp.p')
    os.remove('temp.p')
    return size


def write_model_stats(model, dataloader, file, model_description):
    file.write('\n{}:\n'.format(model_description))
    file.write('size (KB): {}\n'.format(get_model_size(model)))
    accuracy = test(model, dataloader, seed=0, verbose=False)
    file.write('accuracy: {} %\n'.format(accuracy))
    file.flush()


def empty_run(model, data_loader, running_time):
    # switch model to evaluate mode
    model.eval()

    # run on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        start_time = time.time()
        runtime = 0

        while runtime < running_time:
            for features, _ in data_loader:
                features = features.to(device)
                model(features)
                del features, _
            peek_time = time.time()
            runtime = peek_time - start_time


def run_model(model, dataloader, sleep_time, run_time, file, model_description):
    file.write('sleep: {}\n'.format(time.time()))
    file.flush()
    time.sleep(sleep_time)

    file.write('\nrunning {}:\n'.format(model_description))
    file.write('start: {}\n'.format(time.time()))
    file.flush()
    empty_run(model, dataloader, run_time)
    file.write('end: {}\n'.format(time.time()))
    file.flush()
