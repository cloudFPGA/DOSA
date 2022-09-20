from torchvision import transforms

transforms = {
    'cifar10_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ]),
    
    'cifar10_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ]),
    
    'mnist_train': transforms.Compose([
        transforms.ToTensor(),
    ]),

    'mnist_test': transforms.Compose([
        transforms.ToTensor(),
    ])
}
