import torch


def controlled_calibrate(model, data):
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if not isinstance(data, list):
        data = [data]

    for images in data:
        images = images.to(device)
        model(images)


def calibrate(model, test_loader, num_steps=1, seed=None):
    model.eval()

    # run on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    count = 0
    for images, _ in test_loader:
        if count >= num_steps:
            break
        images = images.to(device)
        model(images)
        count += 1


def test(model, test_loader):
    # switch to evaluate mode
    model.eval()

    # run on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
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
