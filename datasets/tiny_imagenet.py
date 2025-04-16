'''
https://www.kaggle.com/datasets/xiataokang/tinyimagenettorch/data
'''
import torch
import torchvision
import torchvision.transforms as transforms
import kaggle
import os


def download_tinyimagenet():
    # Download latest version
    path_to_download = "./data/"
    dataset = "xiataokang/tinyimagenettorch"
    kaggle.api.dataset_download_files(dataset, path=path_to_download, unzip=True)

def load_tinyimagenet(batch_size=64, size=64, n_samples=None):
    # Download
    if not os.path.isdir("./data/tiny-imagenet-200"):
        download_tinyimagenet()

    # Transformations
    if size != 64:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    n_classes = 200

    # Datasets
    trainset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform)
    if n_samples is not None:
        trainset = get_reduced_dataset(trainset, n_samples)
    testset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader, (3, size, size), n_classes
