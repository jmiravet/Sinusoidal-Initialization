import torch
import torch.nn as nn
import torchvision

def resnet50(input_shape, num_classes):
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(2048, num_classes, bias=True)
    return model