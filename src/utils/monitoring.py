import os
import torch
import time


def get_model_size(model):
    torch.save(model.state_dict(), 'temp.p')
    size = os.path.getsize('temp.p')
    os.remove('temp.p')
    return size


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
                outputs = model(features)
                del features, _, outputs
            peek_time = time.time()
            runtime = peek_time - start_time
            print(runtime)
