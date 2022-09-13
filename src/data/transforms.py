from torchvision import transforms

transforms = {
    'cifar10': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ]),

    'cifar100': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ]),

    'mnist': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
}
