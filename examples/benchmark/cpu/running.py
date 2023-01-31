import os.path
import torch
import src
import time
import numpy as np


def empty_run(model, input_data, bs, run_interval):
    with torch.no_grad():
        runtime = 0
        start_time = time.time()

        while runtime < run_interval:
            for i in range(input_data.shape[0]//bs):
                model(input_data[bs*i:bs*(i+1)])
            peek_time = time.time()
            runtime = peek_time - start_time


def get_model_size(model):
    torch.save(model.state_dict(), 'temp.pt')
    size = os.path.getsize('temp.pt')
    os.remove('temp.pt')
    return size


def compute_models_size(models, logger):
    for description, model in models.items():
        logger.write_model_size(description, get_model_size(model))


def compute_models_accuracy(models, dataloader, logger):
    for description, model in models.items():
        accuracy = src.test(model, dataloader, seed=0, verbose=False)
        logger.write_model_accuracy(description, accuracy)


def compute_models_runtime(models, input_shape, batch_sizes, logger, nb_executions=20):
    for description, model in models.items():
        runtimes = []
        for bs in batch_sizes:
            shape = (bs,) + input_shape
            input_data = torch.randn(shape).to('cpu')
            model.eval()
            model.to('cpu')

            # run
            runtime = []
            for _ in range(nb_executions):
                with torch.no_grad():
                    start = time.time()
                    model(input_data)
                    end = time.time()
                    runtime.append((end - start) * 1000)
            runtime = np.asarray(runtime)
            min = np.min(runtime)
            max = np.max(runtime)
            median = np.median(runtime)
            runtimes.append((bs, [min, max, median]))

        logger.write_model_runtimes(description, runtimes)


def empty_run_models(models, input_shape, batch_sizes, logger, sleep_interval, run_interval):
    for description, model in models.items():
        logger.write_model_empty_run(description)
        for bs in batch_sizes:
            shape = (batch_sizes[-1],) + input_shape
            input_data = torch.randn(shape).to('cpu')
            model.eval()
            model.to('cpu')

            # sleep
            logger.write_model_sleep(bs)
            time.sleep(sleep_interval)

            # run
            logger.write_model_start_run(bs)
            empty_run(model, input_data, bs, run_interval)
            logger.write_model_end_run(bs)

