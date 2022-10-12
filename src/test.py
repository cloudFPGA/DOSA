import torch


def test(model, test_loader, neval_batches=None):
    # switch to evaluate mode
    model.eval()

    # run on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        correct = 0
        total = 0
        iteration = 0

        for images, labels in test_loader:
            if neval_batches is not None and ++iteration >= neval_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(total,
                                                                       None if total == 0 else 100 * correct / total),
          flush=True)
