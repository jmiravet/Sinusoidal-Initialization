import torch
import torchvision
import torchvision.transforms as transforms


def load_imagenette(batch_size=64, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    n_classes = 200

    trainset = torchvision.datasets.ImageFolder(root="./data/imagenette2/train", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder(root="./data/imagenette2/val", transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader, (3, size, size), n_classes