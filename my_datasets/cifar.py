import torch
import torchvision
import torchvision.transforms as transforms
import os


def load_cifar10(batch_size=64, size=32, n_samples=None):
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(size, padding=4),  # Data augmentation
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    n_classes = 10

    download = not os.path.isdir("./data/cifar-10-batches-py")
    # Datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=test_transform)
    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, (3, size, size), n_classes

def load_cifar100(batch_size=64, size=32, n_samples=None):
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(size, padding=4),  # Data augmentation
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    n_classes = 100

    download = not os.path.isdir("./data/cifar-100-batches-py")
    # Datasets
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=download, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=download, transform=test_transform)
    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader, (3, size, size), n_classes