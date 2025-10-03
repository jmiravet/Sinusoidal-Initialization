import torch
import torch.nn as nn
import numpy as np

def fernandez_sinusoidal(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        for i in range(n_out):
            l = ((i+1) * 2*np.pi / n_in)
            phase = np.random.rand() * 2 * np.pi
            i_position = position * l + phase 
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
            phase = np.random.rand() * 2 * np.pi
            i_position = position * l + phase 
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal3(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in) * (2 * np.pi / n_in) + (2 * np.pi / n_out)

        for i in range(n_out):
            weight[i,:] = np.sin(position * (i+1))
        
        var = np.var(weight, ddof=1)
        amplitude = np.sqrt(2/ (var * (n_out+n_in)))
        weight = weight * amplitude

        weight = torch.from_numpy(weight)
        return weight

    def initialize_conv(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        phases = np.arange(c_out) * (2 * np.pi / c_out)

        for i in range(c_out):
            l = ((i+1) * 2*np.pi / c_out)
            i_position = position * l + phases[i]
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal2(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        for i in range(n_out):
            l = ((i+1) * 2*np.pi / n_in)
            phase = np.random.rand() * 2 * np.pi
            i_position = position * l + phase 
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
        position = np.arange(c_out) 
        for i in range(n_in):
            l = ((i+1) * 2*np.pi / n_in)
            phase = np.random.rand() * 2 * np.pi
            i_position = position * l + phase 
            weight[:,i] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal4(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in)
        phases = np.arange((n_out // 2)) * (2 * np.pi / (n_out // 2))

        for i in range(n_out):
            l = ((i//2)+1) * 2*np.pi / (n_in)
            i_position = position * l + phases[(i//2)] + ((i%2) * np.pi / 2)
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        amplitude = np.sqrt(2/(var * (n_out+n_in)))
        weight = weight * amplitude

        weight = torch.from_numpy(weight)
        return weight

    def initialize_conv(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        phases = np.arange((c_out // 2)) * (2 * np.pi / (c_out // 2))

        for i in range(c_out):
            l = ((i+1) * 2*np.pi / c_out)
            i_position = position * l + phases[(i//2)] + ((i%2) * np.pi / 2)
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal5(module, mode="all"):
    def initialize_linear(weight, mode):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in)

        for i in range(0,n_out):
            i_position = ((position+i) % n_in) - (n_in-1) / 2
            weight[i,:] = np.sin(np.pi * (i+1)  *(i_position/n_in))
        
        var = np.var(weight, ddof=1)
        if mode == "fan_in":
            amplitude = np.sqrt(2/ (var * (n_in)))
        elif mode == "fan_out":
            amplitude = np.sqrt(2/ (var * (n_out)))
        else:
            amplitude = np.sqrt(2/ (var * (n_out+n_in)))

        weight = weight * amplitude

        weight = torch.from_numpy(weight)
        return weight

    def initialize_conv(weight, mode):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 

        for i in range(c_out):
            i_position = ((position+i) % n_in) - (n_in-1) / 2
            weight[i,:] = np.sin(np.pi * (i+1)  *(i_position/n_in))
        
        var = np.var(weight, ddof=1)
        if mode == "fan_in":
            amplitude = np.sqrt(2/ (var * (n_in)))
        elif mode == "fan_out":
            amplitude = np.sqrt(2/ (var * (c_out)))
        else:
            amplitude = np.sqrt(2/ (var * (c_out+n_in)))
        weight = weight * amplitude
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight, mode)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight, mode)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal_random(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in) * (2 * np.pi / n_in) + (2 * np.pi / n_out)

        for i in range(n_out):
            weight[i,:] = np.sin(position * (i+1))
        
        var = np.var(weight, ddof=1)
        amplitude = np.sqrt(2/ (var * (n_out+n_in)))
        weight = weight * amplitude

        weight = torch.from_numpy(weight)
        idx = torch.randperm(weight.shape[0])
        weight = weight[idx].view(weight.size())
        return weight

    def initialize_conv(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        phases = np.arange(c_out) * (2 * np.pi / c_out)

        for i in range(c_out):
            l = ((i+1) * 2*np.pi / c_out)
            i_position = position * l + phases[i]
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        idx = torch.randperm(weight.shape[0])
        weight = weight[idx].view(weight.size())
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def orthogonal(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def xavier_normal(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def xavier_uniform(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def kaiming_normal(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def kaiming_uniform(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def default_initialization(module):
    pass

def reset_parameters(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()

def zero(module):
    import math
    from scipy.linalg import hadamard
    def ZerO_Init_on_matrix(matrix_tensor):
        # Algorithm 1 in the paper.
        
        m = matrix_tensor.size(0)
        n = matrix_tensor.size(1)
        
        if m <= n:
            init_matrix = torch.nn.init.eye_(torch.empty(m, n))
        elif m > n:
            clog_m = math.ceil(math.log2(m))
            p = 2**(clog_m)
            init_matrix = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))
        
        return init_matrix

    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = ZerO_Init_on_matrix(module.weight.data)
