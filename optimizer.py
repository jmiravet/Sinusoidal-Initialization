import torch
import torch.nn as nn
import numpy as np

class NeuronBalancerWrapper(torch.optim.Optimizer):
    def __init__(self, base_optimizer):
        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise ValueError("base_optimizer must be an instance of torch.optim.Optimizer")

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Fully connected layer (Linear): shape (out_features, in_features)
                if p.data.dim() == 2:
                    mean_per_neuron = p.data.mean(dim=1, keepdim=True)
                    p.data -= mean_per_neuron

                # Conv2d layer: shape (out_channels, in_channels, kH, kW)
                elif p.data.dim() == 4:
                    # Compute mean for each output channel
                    mean_per_kernel = p.data.mean(dim=(1, 2, 3), keepdim=True)
                    p.data -= mean_per_kernel
                    
    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)