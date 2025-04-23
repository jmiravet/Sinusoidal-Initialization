import torch
import torch.nn as nn
import numpy as np

def initialize_linear(weight):
    # A: amplitud, l: longitud de onda, d: desplazamiento
    n_out, n_in = weight.shape
    weight = np.empty((n_out, n_in), dtype=np.float32)
    position = np.arange(n_in) 
    for i in range(n_out):
        l = ((i+1) * 2*np.pi / n_in)
        d = np.random.rand() * 2 * np.pi
        i_position = position * l + d
        weight[i,:] = np.sin(i_position)
    
    var = np.var(weight, ddof=1)
    a = np.sqrt(2/(var * (n_out+n_in)))
    weight = weight * a

    weight = torch.from_numpy(weight)
    return weight

def initialize_conv(weight):
    # A: amplitud, l: longitud de onda, d: desplazamiento
    c_out, c_in, h, w = weight.shape
    n_in = c_in*h*w
    weight = np.empty((c_out, n_in), dtype=np.float32)
    position = np.arange(n_in) 
    for i in range(c_out):
        l = ((i+1) * 2*np.pi / c_out)
        d = np.random.rand() * 2 * np.pi
        i_position = position * l + d
        weight[i,:] = np.sin(i_position)
    
    var = np.var(weight, ddof=1)
    a = np.sqrt(2/(var * (n_in+c_out)))
    weight = weight * a
    weight = weight.reshape((c_out, c_in, h, w))

    weight = torch.from_numpy(weight)
    return weight

def initialize_conv2(weight):
    # A: amplitud, l: longitud de onda, d: desplazamiento
    c_out, c_in, h, w = weight.shape
    n_in = c_in*h*w
    weight = np.empty((c_out, n_in), dtype=np.float32)
    position = np.arange(c_out) 
    for i in range(n_in):
        l = ((i+1) * 2*np.pi / n_in)
        d = np.random.rand() * 2 * np.pi
        i_position = position * l + d
        weight[:,i] = np.sin(i_position)
    
    var = np.var(weight, ddof=1)
    a = np.sqrt(2/(var * (n_in+c_out)))
    weight = weight * a
    weight = weight.reshape((c_out, c_in, h, w))

    weight = torch.from_numpy(weight)
    return weight

def fernandez_sinusoidal(module):
    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal2(module):
    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv2(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def default_initialization(module):
    pass
