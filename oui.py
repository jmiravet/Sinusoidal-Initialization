import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import types
from torch.autograd.functional import jacobian
import itertools, random

from common import *
from initialize import *
from lsuv import lsuv_with_dataloader
from my_datasets.cifar import load_cifar100
from models.lenet import load_lenet5
from my_datasets.imagenette import load_imagenette as load_dataset
from models.lenet import load_lenet5
from models.mlp import load_mlp
from models.vit import load_vit as load_model

SINGVALS = []
optimizer = optim.AdamW
EPOCHS = 10
LR = 0.001

class CaptureLayer(torch.nn.Module):
    def __init__(self, layer):
        super(CaptureLayer, self).__init__()
        self.layer = layer

    def forward(self, input):
        output = self.layer(input)
        self.captured_output = output > 0
        return output

def convert_model(model):
    layers_to_capture = []
    layers_names = []
    for key, module in model._modules.items():
        if type(module) is CaptureLayer:
            pass
        elif type(module) is nn.ReLU:
            new_relu = CaptureLayer(nn.ReLU(inplace=module.inplace))
            layers_to_capture += [new_relu]
            setattr(model, key, new_relu)
            layers_names += [key]
        elif type(module) is nn.SiLU:
            new_silu = CaptureLayer(nn.SiLU(inplace=module.inplace))
            layers_to_capture += [new_silu]
            setattr(model, key, new_silu)
            layers_names += [key]
        elif type(module) is nn.GELU:
            new_gelu = CaptureLayer(nn.GELU())
            layers_to_capture += [new_gelu]
            setattr(model, key, new_gelu)
            layers_names += [key]
        else:
            _, new_layers_to_capture, new_layers_names = convert_model(module)
            if len(new_layers_to_capture) > 0:
               layers_to_capture += [*new_layers_to_capture]
               for l in new_layers_names:
                   layers_names += [f"{key}.{l}"]
    return model, layers_to_capture, layers_names

def conver_model(model):
    layers

def CaptureModel(model):
    model, layers_to_capture, layers_names = convert_model(model)
    model.layers_to_capture = layers_to_capture
    model.layers_names = layers_names
    return model

def on_batch_OUI(model, k = 8*7//2):
    if not hasattr(model, "oui_comb"):
        num_rows = model.layers_to_capture[0].captured_output.shape[0]
        comb = list(itertools.combinations(range(num_rows), 2))
        if k is not None: 
            comb = random.sample(comb, np.min([len(comb),k]))
        else:
            k = len(comb)
        model.oui_comb = torch.tensor(comb, dtype=torch.long, device=model.device)
        model.k = k
    
    oui_list = torch.empty((model.k, len(model.layers_to_capture)), device=model.device)
    limit_list = torch.empty(len(model.layers_to_capture), device = model.device)
    
    for l, layer in enumerate(model.layers_to_capture):
        sub_matrix = layer.captured_output.reshape(layer.captured_output.shape[0], -1)
        limit = sub_matrix.shape[1]//2
        hamming_distances = torch.sum(sub_matrix[model.oui_comb[:, 0]] != sub_matrix[model.oui_comb[:, 1]], dim=1)
        oui_list[:, l] = torch.clamp(hamming_distances, max=limit) 
        limit_list[l] = limit
    oui = ( oui_list / limit_list ).mean(dim=0)

    # sigma = torch.std( oui_list / limit_list , dim=0, unbiased=True)
    # print(len(sigma))
    # n = (1.96 / 0.05 * sigma )**2
    # print(f'Valor de n mínimo = {torch.min(n)}, máximo = {torch.max(n)}, medio = {torch.mean(n)}.')
    # exit()
    return oui

def get_activations(model, samples):
    activations = []
    hooks = []
    def forward_hook(module, input, output):
        activations.append(output)
    def add_hooks(module):
        if isinstance(module, nn.Linear): # or isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(forward_hook))
    def remove_hooks():
        for hook in hooks:
            hook.remove()

    model.apply(add_hooks)
    with torch.no_grad():
        model.forward(samples)
    remove_hooks()
    return activations

def percentage_positive(tensor_list):
    percentages = []
    for tensor in tensor_list:
        total_elements = tensor.numel()
        positive_elements = (tensor > 0).sum().item()
        percentage = (positive_elements / total_elements) * 100
        percentages.append(percentage)
    return percentages

def on_batch_APD(activations, k = 8*7//2):
    num_rows = activations[0].shape[0]
    comb = list(itertools.combinations(range(num_rows), 2))
    if k != None: comb = random.sample(comb, np.min([len(comb),k]))

    apd_list = torch.empty((k, len(activations)))
    limit_list = torch.empty(len(activations))
    for l, layer in enumerate(activations):
        sub_matrix = layer
        sub_matrix = sub_matrix.reshape(-1, sub_matrix.shape[-1])
        limit = sub_matrix.shape[1]//2
        hamming_distances = torch.tensor([torch.sum(sub_matrix[i] != sub_matrix[j]).item() for i, j in comb])
        apd_list[:, l] = torch.clip(hamming_distances, None, limit) 
        limit_list[l] = limit
    apd = (apd_list / limit_list).mean(dim=0)
    return apd

def main():
    
    train_loader, val_loader, input_shape, num_classes = load_dataset(batch_size=64, size=224)
    # Create model and initialize weights 
    model = load_model(input_shape, num_classes)
    model.to("cuda")
    model.eval()

    # Initialize
    #model.apply(fernandez_sinusoidal3).to("cuda")
    model.apply(orthogonal).to("cuda")
    #lsuv_with_dataloader(model, train_loader)

    model = CaptureModel(model)
    model.device = torch.device("cuda")
    for i, samples in enumerate(train_loader):
        with torch.no_grad():
            outputs = model(samples)
            batch_oui = on_batch_OUI(model)
            print(batch_oui)
            break


if __name__ == '__main__':
    main()