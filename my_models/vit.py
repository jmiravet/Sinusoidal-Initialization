import torch
import torch.nn as nn
import torchvision

def load_vit(input_shape, num_classes):
    model = torchvision.models.vit_b_16()
    return model