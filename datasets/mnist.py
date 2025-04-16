import torch
import torchvision
import torchvision.transforms as transforms


def load_mnist(batch_size=64, size=28, n_samples=None):
    # Transformations
    if size != 28:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()])
    else:
        transform = transforms.ToTensor()

    n_classes = 10

    # Datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    if n_samples is not None:
        trainset = get_reduced_dataset(trainset, n_samples)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                              num_workers=2, prefetch_factor=100, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=2, prefetch_factor=100, pin_memory=True)
    return trainloader, testloader, (1, size, size), n_classes