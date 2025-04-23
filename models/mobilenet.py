import torch
import torch.nn as nn
import torchvision

def load_mobilenet(input_shape, num_classes):
    model = torchvision.models.mobilenet_v3_small()
    model.classifier[-1] = nn.Linear(1024, num_classes, bias=True)
    return model