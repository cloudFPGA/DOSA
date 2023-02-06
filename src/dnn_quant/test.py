import torch


def calibrate(model, test_loader, num_steps=1, seed=None):
    from dnn_quant.models.quantized.quant_module import QuantModule
    if isinstance(model, QuantModule):
        model.calibrate()
    else:
        model.eval()

    # run on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    count = 0
    for features, _ in test_loader:
        if count >= num_steps:
            break
        features = features.to(device)
        model(features)
        count += 1


def test(model, test_loader, seed=None, verbose=True):
    # switch to evaluate mode
    model.eval()

    # run on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        correct = 0
        total = 0

        if seed is not None:
            torch.manual_seed(seed)
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del features, labels, outputs

    if verbose:
        print('Accuracy of the network on the {} test data: {} %'.format(total,
                                                                         None if total == 0 else 100 * correct / total),
              flush=True)

    return None if total == 0 else 100 * correct / total
