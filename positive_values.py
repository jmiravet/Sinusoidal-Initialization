import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import types
import itertools, random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import jacobian

from common import *
from initialize import *
from my_datasets.cifar import load_cifar100
from models.lenet import load_lenet5

optimizer = optim.AdamW
EPOCHS = 10
LR = 0.001


def get_activations(model, samples):
    activations = []
    hooks = []
    def forward_hook(module, input, output):
        activations.append(output)
    def add_hooks(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(forward_hook))
    def remove_hooks():
        for hook in hooks:
            hook.remove()

    model.apply(add_hooks)
    with torch.no_grad():
        model.forward(samples)
    remove_hooks()
    return activations

def percentage_positive(tensor):
    total_elements = tensor.numel()
    positive_elements = (tensor > 0).sum().item()
    percentage = (positive_elements / total_elements) * 100
    return percentage


def mcnc(tensor):
    pass

def main():
    global SINGVALS
    dataset = load_cifar100
    model = load_lenet5
    initialization = fernandez_sinusoidal4
    #initialization = reset_parameters
    initialization = orthogonal
    initialization = xavier_uniform

    # Load data
    train_loader, test_loader, input_shape, num_classes = dataset(size=32, batch_size=64)

    total_activations = None
    print(initialization.__name__)
    # Create model and initialize weights 
    model = model(input_shape, num_classes)
    # Initialize
    model.apply(initialization)


    total_activations = None
    for i, (samples, labels) in enumerate(train_loader):
        if i == 100 or samples.shape[0] != 64:
            break
        # shape = samples.shape
        # samples = torch.rand(size=shape)
        activations = get_activations(model, samples)
        if total_activations is None:
            total_activations = activations
            for layer in range(len(total_activations)):
                total_activations[layer] = [activations[layer].detach()]
        else:
            for layer in range(len(total_activations)):
                total_activations[layer].append(activations[layer].detach())

    for layer in range(len(total_activations)):
        total_activations[layer] = torch.stack(total_activations[layer], axis=0)
        total_activations[layer] = torch.reshape(total_activations[layer], shape=(-1, *total_activations[layer].shape[2:]))

    percentages = []
    for activation in total_activations:
        percentage = percentage_positive(activation)
        percentages.append(percentage)
        print(f"{percentage:.3f}", end="; ")
    print(f"Mean deviation: {np.mean(np.abs(np.array(percentages)-50)):.3f}")


if __name__ == '__main__':
    main()