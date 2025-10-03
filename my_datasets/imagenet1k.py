import torch
import torchvision
import torchvision.transforms as transforms
import os

from .utils import *

def download_imagenet1k():
    raise Exception("Download Imagenet1k dataset from: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php")
    '''
        wget --user=USERNAME --password=PASSWORD \
        http://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

        wget --user=USERNAME --password=PASSWORD \
        http://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
    '''

def load_imagenet1k(batch_size=64, size=224, n_samples=None):
    '''
    Imagenet 1k dataset images have different sizes, if no resize is in place, the dataloader needs this collate_fn to be passed
    as an argument. More testing needed in this case.
    '''
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    if not os.path.isdir("/scratch/imagenet1k/"):
        download_imagenet1k()

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    num_classes = 1000
    # Datasets
    trainset = torchvision.datasets.ImageNet(root="/scratch/imagenet1k/", split="train", transform=transform)
    if n_samples is not None:
        trainset = get_reduced_dataset(trainset, n_samples)
    testset = torchvision.datasets.ImageNet(root="/scratch/imagenet1k/", split="val", transform=transform)
    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4) #, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4) #, collate_fn=collate_fn)
    
    return trainloader, testloader, (3, size, size), num_classes
