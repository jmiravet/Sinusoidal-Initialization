import torch
import torch.nn as nn
import torchvision

def load_efficientnet(input_shape, num_classes):
    model = torchvision.models.efficientnet_v2_s()
    model.classifier[-1] = nn.Linear(1280, num_classes, bias=True)
    return model