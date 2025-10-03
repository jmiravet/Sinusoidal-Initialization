import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import arcsine
import types
import itertools, random
from scipy.stats import norm 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import jacobian

from common import *
from initialize import *
from lsuv import lsuv_with_dataloader
from my_datasets.cifar import load_cifar100 as load_dataset
#from my_datasets.imagenette import load_imagenette as load_dataset
from models.lenet import load_lenet5
from models.mlp import load_mlp
#from models.vit import load_vit as load_model
from models.resnet50 import load_resnet50 as load_model
from test import *

plt.rcParams.update({
    'text.usetex' : True,
    'font.family': 'Times New Roman',
})
# plt.rcParams['font.family'] = 'Times New Roman'
plot_dir = "./plots/"
optimizer = optim.AdamW
EPOCHS = 10
LR = 0.001

def initialize_arcsine_vector(weight, a=0.0, b=1.0):
    """
    Initialize a vector with values following the arcsine distribution
    on the interval [a, b].

    Parameters:
        size (int): Number of samples to generate.
        a (float): Lower bound of the distribution (default 0.0).
        b (float): Upper bound of the distribution (default 1.0).

    Returns:
        np.ndarray: Vector of samples.
    """
    n_out, n_in = weight.shape
    amplitude = np.sqrt(4/(n_out+n_in))
    u = np.random.uniform(0, 1, n_out*n_in)
    samples = -1 * amplitude + 2 * amplitude * (np.sin(np.pi * u / 2))**2
    weight = samples.reshape((n_out, n_in)).astype(dtype=np.float32)
    
    weight = torch.from_numpy(weight)
    return weight

def random_sinusoidal(module):    
    if isinstance(module, nn.Linear):
        module.weight.data = initialize_arcsine_vector(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def initialize_linear4(weight):
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

def initialize_linear_random(weight):
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

def get_activations(model, layer, samples):
    activations = []
    hooks = []
    def forward_hook(module, input, output):
        activations.append(output.detach().to("cpu"))
    def add_hooks(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(forward_hook))
    def remove_hooks():
        for hook in hooks:
            hook.remove()

    model.eval()
    #model.apply(add_hooks)
    # model.encoder.layers.encoder_layer_0.mlp[1] = nn.ReLU()
    hooks.append(layer.register_forward_hook(forward_hook))
    # hooks.append(model.layer4[2].relu.register_forward_hook(forward_hook))
    model.to("cuda")
    with torch.no_grad():
        model.forward(samples.to("cuda"))
    remove_hooks()
    return activations

def percentage_positive(tensor):
    total_elements = tensor.numel()
    positive_elements = (tensor > 0).sum().item()
    percentage = (positive_elements / total_elements) * 100
    return percentage

def plot_distribution_separate(tensor):
    fig, axs = plt.subplots(1, tensor.shape[0], tight_layout=True)

    colors = plt.colormaps['viridis_r'](np.linspace(0, 1, tensor.shape[0]))
    for i, x in enumerate(tensor):
        axs[i].hist(x, bins=100) #, color=colors[i])
    plt.show()

def plot_distribution(tensor):
    plt.figure()
    plt.hist(tensor, bins=200)
    plt.show()

def test_plot(tensor):
    plt.figure()
    colors = plt.colormaps['viridis_r'](np.linspace(0, 1, tensor.shape[0]))
    for i, x in enumerate(tensor):
        plt.plot(x, color=colors[i])
    plt.show()

def on_batch_OUI(tensor, k = 8*7//2):
    num_rows = tensor.shape[0]
    comb = list(itertools.combinations(range(np.min([num_rows, 64])), 2))
    comb = random.sample(comb, np.min([len(comb),k]))
    oui_comb = torch.tensor(comb, dtype=torch.long, device="cpu")
        
    sub_matrix = tensor
    limit = sub_matrix.shape[1]//2
    hamming_distances = np.sum(sub_matrix[oui_comb[:, 0]] != sub_matrix[oui_comb[:, 1]], axis=1)
    oui_list = np.clip(hamming_distances, a_min=None, a_max=limit) 
    limit_list = limit
    oui = np.mean(( oui_list / limit_list ))

    # sigma = torch.std( oui_list / limit_list , dim=0, unbiased=True)
    # print(len(sigma))
    # n = (1.96 / 0.05 * sigma )**2
    # print(f'Valor de n mínimo = {torch.min(n)}, máximo = {torch.max(n)}, medio = {torch.mean(n)}.')
    # exit()
    return oui

def main():

    if False: # Mas viciosa
        input_shape = (1000000, 1000)
        samples = torch.empty(size=input_shape, device="cuda")
        weights = torch.empty(size=(input_shape[-1],input_shape[-1]), device="cuda")
        relu = nn.ReLU()
        bn = nn.BatchNorm1d(1000)
        bn = bn.to("cuda")

        plt.figure()

        with torch.no_grad():
            samples = samples.normal_(mean=0, std=1)
            #samples = samples.uniform_(-0.5, 0.5)
            #samples = bn.forward(samples)
            samples = relu.forward(samples)
            #samples = bn.forward(samples)
            weights = nn.init.kaiming_uniform_(weights, mode="fan_in").T
            #weights = initialize_linear4(weights).to("cuda").T
            #weights = initialize_arcsine_vector(weights).to("cuda").T
            winner = torch.argmax(torch.abs(torch.mean(weights, dim=0)))
            weights = weights[:,winner:winner+1]
            #weights = weights[:,0:0+1]
            output = torch.matmul(samples, weights)
        
        output = output.to("cpu").flatten().detach().numpy()
        mean = np.mean(output, axis=0)
        std = np.std(output, axis=0)
        print(mean, std)
        
        mu  = mean        # mean     (change me)
        sigma = std      # std‐dev  (change me)
        x_min, x_max = mu - 4*sigma, mu + 4*sigma
        x = np.linspace(x_min, x_max, 500)
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        plt.hist(output, bins=200, density=True)
        plt.plot(x, pdf, lw=2, label=f"μ={mu}, σ={sigma}")
        plt.show()
        exit()
        plot_distribution(output)
        exit()
    
    if False: # Histograma y porcentaje viciadas
        samples_indexes = None
        sinthetic = True
        histogram = True
        for row in range(4):
            for collumn in range(3):
                init_name, initialization = "Glorot", xavier_normal
                n_samples = 10000
                features = 2000
                samples = torch.empty(size=(n_samples, features), device="cuda")
                samples = samples.normal_(mean=0, std=1)
                val_loader = torch.utils.data.DataLoader(samples, batch_size=64, shuffle=False, drop_last=True)
                # Create model and initialize weights
                model = nn.Sequential(
                    nn.Hardswish(),
                    nn.Linear(features, features, bias=False),
                    nn.Hardswish(),
                    nn.Linear(features, features, bias=False),
                    nn.Hardswish(),
                    nn.Linear(features, features, bias=False),
                )
                output_layer = model[collumn*2+1]
                # print(model)
                # exit()
                # Initialize
                model.apply(initialization)
                for i in range(row):
                    model[i*2+1].apply(fernandez_sinusoidal3)
                
                activations = []
                for i, samples in enumerate(val_loader):
                    if i == 30:
                        break
                    #samples = samples.normal_(mean=0, std=1)
                    batch_activations = get_activations(model, output_layer, samples)
                    activations.append(batch_activations[0].detach().to("cpu").numpy())

                activations = np.stack(activations, axis=0)
                activations = np.reshape(activations, shape=(-1, activations.shape[-1]))

                # Percentage
                activations = activations > 0
                mean = np.mean(activations, axis=0)
                std = np.std(activations, axis=0)
                alpha = 0.3 # entre [0, 0.5] # si queremos mean - 1 std -> alpha = 0.341
                skewed_3 = np.mean(np.abs(norm.cdf(mean / std) - 1/ 2) > alpha)
                
                # Histogram
                if histogram:
                    weights = output_layer.weight.detach().to("cpu").numpy().T
                    sum_weights = np.sum(weights, axis=0)
                    skewed_neurons = np.abs(norm.cdf(mean / std) - 1/ 2) > alpha
                    not_skewed_neurons = np.abs(norm.cdf(mean / std) - 1/ 2) <= alpha
                    skewed_neurons = sum_weights[skewed_neurons]
                    not_skewed_neurons = sum_weights[not_skewed_neurons]

                    bin_size = 0.2  
                    bins = np.arange(start=(-1 * bin_size / 2 + (min(sum_weights) // bin_size) * bin_size), stop=max(sum_weights) + bin_size, step=bin_size)
                    plt.figure(figsize=(4, 3))
                    plt.hist([skewed_neurons, not_skewed_neurons],
                            bins=bins,
                            stacked=True, density=True,
                            label=['Skewed ($\\alpha=0.3$)', 'Not Skewed'],
                            edgecolor="black")
                    plt.yticks([])
                    plt.xlabel('Sum of Weights')
                    plt.ylabel('Density')
                    title = init_name if collumn + 1 > row else "Sinusoidal"
                    plt.title(f"{title} Layer {collumn+1}")
                    plt.xlim(-4.5,4.5)
                    #plt.legend()
                    # plt.savefig(f'{plot_dir}appendix_skewedhistogram_{row+1}_{collumn+1}.pdf', 
                    #             format='pdf', dpi=300,
                    #             bbox_inches='tight')
        
        if histogram:
            plt.show()
        exit()

    if False: # Histograma viciadas
        # for title, initialization in [("Glorot - Normal", xavier_normal),
        #                               ("Glorot - Uniform", xavier_uniform),
        #                               ("Sinusoidal", fernandez_sinusoidal3),
        #                               ("Sinusoidal - Random", random_sinusoidal),
        #                               ("He - Normal", kaiming_normal),
        #                               ("He - Uniform", kaiming_uniform),
        #                               ("Default", default_initialization),
        #                               ("Orthogonal", orthogonal),
        #                               ("LSUV", lsuv_with_dataloader)]:
        
        # for title, initialization in [("LSUV", lsuv_with_dataloader),]:
        for title, initialization in [("Sinusoidal", fernandez_sinusoidal3),]:
            input_shape = (10000, 10000)
            samples = torch.empty(size=input_shape, device="cuda")
            features = input_shape[-1]
            model = nn.Sequential(
                nn.ReLU(),
                nn.Linear(features, features, bias=False),
            )
            model.eval()
            model.to("cuda")

            with torch.no_grad():
                samples = samples.normal_(mean=0, std=1)
                # Initialize
                if title == "LSUV":
                    train_loader = torch.utils.data.DataLoader(samples, batch_size=50, shuffle=False)
                    model = lsuv_with_dataloader(model, train_loader, device="cuda", verbose=False)
                else:
                    model.apply(initialization).to("cuda")
                # model[1].apply(fernandez_sinusoidal3).to("cuda")
                # output = model[0].forward(samples)
                # output = model[1].forward(output)
                # output = model[2].forward(output)
                # output = model[3].forward(output)
                output = model.forward(samples)
            
            weights = model[1].weight.detach().T
            sum_weights = torch.sum(weights, dim=0).to("cpu").numpy()
            output = output.detach().to("cpu").numpy()
            mean = np.mean(output, axis=0)
            std = np.std(output, axis=0)
            
            # Percentage         
            alpha = 0.1 # entre [0, 0.5] # si queremos mean - 1 std -> alpha = 0.341
            skewed_1 = np.mean(np.abs(norm.cdf(mean / std) - 1/ 2) > alpha)
            alpha = 0.3 # entre [0, 0.5] # si queremos mean - 1 std -> alpha = 0.341
            skewed_3 = np.mean(np.abs(norm.cdf(mean / std) - 1/ 2) > alpha)
            print(f"{title:<20}; a=0.1; {skewed_1*100:4.1f}%; a=0.3; {skewed_3*100:4.1f}%", flush=True)


            alpha = 0.3 # entre [0, 0.5] # si queremos mean - 1 std -> alpha = 0.341
            skewed_neurons = np.abs(norm.cdf(mean / std) - 1/ 2) > alpha
            not_skewed_neurons = np.abs(norm.cdf(mean / std) - 1/ 2) <= alpha
            skewed_neurons = sum_weights[skewed_neurons]
            not_skewed_neurons = sum_weights[not_skewed_neurons]

            bin_size = 0.2  
            bins = np.arange(start=(-1 * bin_size / 2 + (min(sum_weights) // bin_size) * bin_size), stop=max(sum_weights) + bin_size, step=bin_size)
            # plt.figure(figsize=(2.5, 2.5))
            plt.figure(figsize=(3, 2))
            plt.hist([skewed_neurons, not_skewed_neurons],
                    bins=bins,
                    stacked=True, density=True,
                    label=['Skewed ($\\alpha=0.3$)', 'Not Skewed'],
                    edgecolor="black")
            plt.yticks([])
            plt.xlabel('Sum of Weights')
            plt.ylabel('Density')
            plt.title(title)
            plt.xlim(-4.5,4.5)
            # plt.legend()
            # plt.savefig(f'{plot_dir}skewedhistogram_{title.lower()}_3.pdf', 
            #             format='pdf', dpi=300,
            #             bbox_inches='tight')
        plt.show()
        exit()

    if False: # Histograma y porcentaje viciadas
        '''
        ResNet(
            (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            (layer1): Sequential(
                (0): Bottleneck(
                (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (downsample): Sequential(
                    (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        '''
        samples_indexes = None
        sinthetic = True
        histogram = False
        convolutional = True
        experiments = [("Glorot - Normal", xavier_normal),
                       ("Glorot - Uniform", xavier_uniform),
                       ("Sinusoidal", fernandez_sinusoidal3),
                       ("Sinusoidal - Random", random_sinusoidal),
                       ("He - Normal", kaiming_normal),
                       ("He - Uniform", kaiming_uniform),
                       ("Default", default_initialization),
                       ("Orthogonal", orthogonal),
                       ("LSUV", lsuv_with_dataloader)]
        experiments = [("Sinusoidal", fernandez_sinusoidal5),]
        # experiments = [("Glorot - Normal", xavier_normal),
        #                ("Sinusoidal", fernandez_sinusoidal5),]
        for title, initialization in experiments:
            
            for i in range(20):
                if sinthetic:
                    if convolutional:
                        samples = 2000
                        c_in, c_out, h, w, kh, kw = 4, 64, 32, 32, 3, 3
                        in_features, out_features = 10, 10
                        samples = torch.empty(size=(samples, c_in, h, w), device="cuda")
                        samples = samples.normal_(mean=0, std=1)
                        val_loader = torch.utils.data.DataLoader(samples, batch_size=64, shuffle=False, drop_last=True)
                        model = nn.Sequential(
                            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(kh, kw), padding="same"),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(kh, kw), padding="same"),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(kh, kw), padding="same"),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(kh, kw), padding="same"),
                        )
                        try:
                            output_layer = model[i*2]
                        except:
                            break
                    else:
                        n_samples = 2000
                        features = 1000
                        samples = torch.empty(size=(n_samples, features), device="cuda")
                        samples = samples.normal_(mean=0, std=1)
                        val_loader = torch.utils.data.DataLoader(samples, batch_size=64, shuffle=False)
                else:
                    train_loader, val_loader, input_shape, num_classes = load_dataset(batch_size=16, size=224)
                # Create model and initialize weights
                if sinthetic: 
                    if not convolutional:
                        model = nn.Sequential(
                            nn.Linear(features, features, bias=False),
                            nn.ReLU(),
                            nn.BatchNorm1d(features),
                            nn.Linear(features, features, bias=False),
                            nn.ReLU(),
                            nn.BatchNorm1d(features),
                            nn.Linear(features, features, bias=False),
                            nn.ReLU(),
                            nn.BatchNorm1d(features),
                            nn.Linear(features, features, bias=False),
                            nn.ReLU(),
                            nn.BatchNorm1d(features),
                            nn.Linear(features, features, bias=False),
                            nn.ReLU(),
                            nn.BatchNorm1d(features),
                            nn.Linear(features, features, bias=False),
                        )
                        try:
                            output_layer = model[i*3+2]
                        except:
                            break
                else:
                    model = load_model(input_shape, num_classes)
                    output_layer = [model.conv1, model.layer1[0].conv1, model.layer1[1].conv1, model.layer1[2].conv1, 
                                    model.layer2[0].conv1, model.layer2[1].conv1, model.layer2[2].conv1,
                                    model.layer3[0].conv1, model.layer3[1].conv1, model.layer3[2].conv1][i]
                    # output_layer = model.layer2[0].conv3
                # exit()
                # Initialize
                if title == "LSUV":
                    model = lsuv_with_dataloader(model, val_loader, device="cuda", verbose=False)
                else:
                    model.apply(initialization)
                activations = []
                for i, samples in enumerate(val_loader):
                    if not sinthetic:
                        samples, labels = samples
                    if i == 30:
                        break
                    # samples = samples.normal_(mean=0, std=1)
                    batch_activations = get_activations(model, output_layer, samples)
                    activations.append(batch_activations[0].detach().to("cpu").numpy())
        
                activations = np.stack(activations, axis=0)
                if sinthetic:
                    if convolutional:
                        activations = np.reshape(activations, shape=(-1, *activations.shape[2:]))
                        activations = np.transpose(activations, axes=(0,2,3,1))
                        activations = np.reshape(activations, shape=(-1, activations.shape[-1]))
                    else:
                        activations = np.reshape(activations, shape=(-1, activations.shape[-1]))
                else:
                    activations = np.reshape(activations, shape=(-1, *activations.shape[2:]))
                    if len(activations.shape) == 4:
                        activations = np.transpose(activations, axes=(0,2,3,1))
                        activations = np.reshape(activations, shape=(-1, activations.shape[-1]))

                # Percentage
                mean = np.mean(activations, axis=0)
                std = np.std(activations, axis=0)   
                weights = output_layer.weight.detach().to("cpu").numpy()
                weights = weights.reshape(weights.shape[0], -1)
                print(f"Weights max mean: {np.max(np.abs(np.mean(weights, axis=1)))}; mean: {np.mean(mean)}, std: {np.std(mean)}", end="  ")         
                alpha = 0.1 # entre [0, 0.5] # si queremos mean - 1 std -> alpha = 0.341
                skewed_1 = np.mean(np.abs(norm.cdf(mean / std) - 1/ 2) > alpha)
                alpha = 0.3 # entre [0, 0.5] # si queremos mean - 1 std -> alpha = 0.341
                skewed_3 = np.mean(np.abs(norm.cdf(mean / std) - 1/ 2) > alpha)
                
                # Histogram
                if histogram:
                    weights = output_layer.weight.detach().to("cpu").numpy()
                    if len(weights.shape) == 2:
                        weights = np.transpose(weights, axes=(1,0))
                    elif len(weights.shape) == 4:
                        weights = np.transpose(weights, axes=(1,2,3,0))
                        weights = np.reshape(weights, shape=(-1, weights.shape[-1]))

                    sum_weights = np.sum(weights, axis=0)
                    skewed_neurons = np.abs(norm.cdf(mean / std) - 1/ 2) > alpha
                    not_skewed_neurons = np.abs(norm.cdf(mean / std) - 1/ 2) <= alpha
                    skewed_neurons = sum_weights[skewed_neurons]
                    not_skewed_neurons = sum_weights[not_skewed_neurons]

                    bin_size = 0.2  
                    bins = np.arange(start=min(sum_weights), stop=max(sum_weights) + bin_size, step=bin_size)
                    plt.figure(figsize=(5, 3))
                    plt.hist([skewed_neurons, not_skewed_neurons],
                            bins=bins,
                            stacked=True, density=True,
                            label=['Skewed ($\\alpha=0.3$)', 'Not Skewed'],
                            edgecolor="black")
                    plt.yticks([])
                    plt.xlabel('Sum of Weights')
                    plt.ylabel('Density')
                    plt.title(title)
                    plt.xlim(-4.5,4.5)
                    plt.legend()
                    # plt.savefig(f'{plot_dir}skewedhistogram_test{alpha*10:.0f}.pdf', 
                    #             format='pdf', dpi=300,
                    #             bbox_inches='tight')

                # OUI
                activations = activations > 0
                n_samples, m_neurons = activations.shape
                m_neurons = 250
                if samples_indexes is None and m_neurons < np.min(activations.shape):
                    samples_indexes = np.random.choice(np.arange(0, np.min(activations.shape)), size=m_neurons, replace=False)
                if samples_indexes is not None:
                    activations = activations[samples_indexes][:,samples_indexes]
                oui = on_batch_OUI(activations)
                print(f"{title:<20}; a=0.1; {skewed_1*100:4.1f}%; a=0.3; {skewed_3*100:4.1f}%; OUI; {oui:4.3f}", flush=True)
        
        if histogram:
            plt.show()
        exit()

    if False: # % Viciosas
        for title, initialization in [("Glorot - Normal", xavier_normal),
                                      ("Sinusoidal", fernandez_sinusoidal3),]:
            input_shape = (10000, 1000)
            #input_shape = (100, 100)
            samples = torch.empty(size=input_shape, device="cuda")
            features = input_shape[-1]
            weights = torch.empty(size=(input_shape[-1],input_shape[-1]), device="cuda")
            relu = nn.ReLU()
            model = nn.Sequential(
                nn.ReLU(),
                nn.Linear(features, features, bias=False),
                nn.ReLU(),
                nn.Linear(features, features, bias=False),
                nn.ReLU(),
                nn.Linear(features, features, bias=False),
            )
            model.to("cuda")
            model.eval()
            samples = samples.normal_(mean=0, std=1)
            train_loader = torch.utils.data.DataLoader(samples, batch_size=64, shuffle=False)

            with torch.no_grad():
                #model.apply(kaiming_normal)
                #model.apply(orthogonal)
                model.apply(initialization).to("cuda")
                #lsuv_with_dataloader(model, train_loader, device=torch.device("cuda"))
                output = model.forward(samples)
                # samples = relu.forward(samples)
                # #weights = nn.init.kaiming_normal_(weights, mode="fan_in").T
                # weights = nn.init.orthogonal_(weights)
                #weights = initialize_linear(weights).to("cuda").T
                # output = torch.matmul(samples, weights)

            # model[1].apply(fernandez_sinusoidal3).to("cuda")
            # model[3].apply(fernandez_sinusoidal3).to("cuda")
            # model[5].apply(fernandez_sinusoidal3).to("cuda")
            
            output = output.to("cpu").detach().numpy()
            mean = np.mean(output, axis=0)
            std = np.std(output, axis=0)
            
            alpha = 0.1 # entre [0, 0.5] # si queremos mean - 1 std -> alpha = 0.341
            viciadas = np.mean(np.abs(norm.cdf(mean / std) - 1/ 2) > alpha)
            print(f"Alpha: {alpha}; {viciadas*100:.2f}%")
            alpha = 0.3 # entre [0, 0.5] # si queremos mean - 1 std -> alpha = 0.341
            viciadas = np.mean(np.abs(norm.cdf(mean / std) - 1/ 2) > alpha)
            print(f"Alpha: {alpha}; {viciadas*100:.2f}%")
        exit()

    if False: # U shaped distribution
        input_shape = (10, 10000)
        weights = torch.empty(size=(input_shape[-1],input_shape[-1]))
        #weights = initialize_linear(weights)
        weights = initialize_arcsine_vector(weights)
        weights = weights.flatten().detach().numpy()
        amplitude = np.max(weights)

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.hist(weights, bins=30, density=True)
        x = np.linspace(0, 1, 400)
        x_plot = (x - 0.5) * amplitude*2
        ax.plot(x_plot, arcsine.pdf(x) * 35,         # scale PDF to histogram counts
                 lw=2, linestyle="--", color='r', label='arcsine distribution')
        
        #ax.text(0, 50, "Arcsine probability density function", horizontalalignment="center", color="r")
        
        ax.set_xticks(ticks=[-1*amplitude, 0, amplitude], labels=["$-a$", 0, "$+a$"])
        ax.set_yticks([])
        ax.set_ylabel("Weight Density")
        ax.set_title("Sinusoidal - Random")
        ax.set_ylim(0, 130)

        fig.savefig(f'{plot_dir}weight_distribution_random.pdf', 
                    format='pdf', dpi=300, )
                    # bbox_inches='tight', )
        plt.show()
        exit()

    if True: # COLORMAP
        weights = torch.empty(size=(101,101))
        weights = initialize_linear_random(weights)
        #weights = initialize_arcsine_vector(weights).to("cuda")
        weights = weights.detach().numpy()
        amplitude = np.max(weights)
        #weights = weights[:-1,:-1]

        fig, ax = plt.subplots(figsize=(3, 2))
        cmap = plt.get_cmap('bwr')  # Good diverging colormap: blue-white-red
        img = ax.imshow(weights, cmap=cmap, vmin=np.min(weights), vmax=np.max(weights),
                        aspect = "equal")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.1, pad=0.1)

        cbar = fig.colorbar(img, cax=cax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Output feature")
        ax.set_ylabel("Input feature")
        #ax.set_title("Weight visualization")

        #fig.savefig(f'{plot_dir}weight_visualization.pdf', 
        #            format='pdf', dpi=300, )
                    # bbox_inches='tight')
        plt.show()
        exit()

    batch_size = 50
    # input_shape, num_classes = (300,), 300
    # input_shape, num_classes = (3, 32, 32), 100
    total_activations = None
    train_loader, _, input_shape, num_classes = load_dataset(batch_size=32, size=224)
    # Create model and initialize weights 
    model = load_model(input_shape, num_classes)
    # Initialize
    model.apply(initialization)

    if False: # QR BARCODE
        total_activations = None
        for i in range(10):
        #for i, (samples, labels) in enumerate(train_loader):
            # if i == 2:
            #     break
            # shape = samples.shape
            samples = torch.empty(size=(batch_size, *input_shape))
            #samples = samples.uniform_(from_=-0.5, to=0.5)
            samples = samples.normal_(mean=0, std=1)
            activations = get_activations(model, samples)
            if total_activations is None:
                total_activations = activations
                for layer in range(len(total_activations)):
                    total_activations[layer] = [activations[layer].detach().numpy()]
            else:
                for layer in range(len(total_activations)):
                    total_activations[layer].append(activations[layer].detach().numpy())
 
        total_activations = total_activations[2:]
        layer_limits = [-0.5,]
        for layer in range(len(total_activations)):
            activations = total_activations[layer]
            activations = np.stack(activations, axis=0)
            activations = np.reshape(activations, shape=(-1, *activations.shape[2:]))
            activations = np.reshape(activations, shape=(activations.shape[0], -1))
            #activations = np.transpose(activations)
            total_activations[layer] = activations <= 0
            print(activations.shape)
            layer_limits.append(activations.shape[-1] + layer_limits[-1])
      

        widths = [layer.shape[-1] for layer in total_activations]

        fig, axs = plt.subplots(
            1,                          # one row
            len(total_activations),     # one column per layer
            sharex=False, sharey=False,
            gridspec_kw={
                'width_ratios': widths, # <- goes here
                'wspace': 0.1,            # no horizontal padding
                'hspace': 0             # (does nothing for a single row, but harmless)
            },
            figsize=(5, 3)    # optional: make canvas wide enough but keep height fixed
        )

        cmap = plt.get_cmap('binary')  # 0 blanco, 1 negro
        #cmap = plt.get_cmap('viridis')
        for i in range(len(axs)):
            tensor = total_activations[i]

            img = axs[i].imshow(tensor, cmap=cmap, vmin=0, vmax=1,
                                        aspect="equal")
            
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xlabel(f"Layer {i+1}")
        
        axs[0].set_ylabel("Samples")
        #axs[1].set_title("Xavier normal intialization neuron activation state")
        axs[1].set_title("Sinusoidal")

        # for x in layer_limits[1:-1]:
        #     ax.vlines(x, ymin=-0.5, ymax=tensor.shape[0]-0.5, colors="b", linestyles="--", alpha=1)

        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size=0.2, pad=0.1)

        # ax.tick_params(axis='x', which="major", color='b', length=0, direction='in', width=0)
        # ax.tick_params(axis='x', which="minor", color='b', length=12, direction='out', width=1)
        # ax.set_xticks(layer_limits[1:-1], [], minor=True)
        #axs.set_xticks(major_ticks, ["Layer 1", "Layer 2", "Layer 3"], minor=False)
        #axs.set_yticks([])
        # axs.set_xlabel("Neurons")
        # axs.set_ylabel("Samples")
        #fig.suptitle("Neuron activation state")

        fig.savefig(f'{plot_dir}activations_{initialization.__name__}.pdf', 
                    format='pdf', dpi=300,
                    bbox_inches='tight')
        plt.show()
        exit()

    if False: # QR BARCODE Single layer, real data
        samples_indexes = None
        for title, initialization in [("Glorot - Normal", xavier_normal),
                                      ("Glorot - Uniform", xavier_uniform),
                                      ("Sinusoidal", fernandez_sinusoidal3),
                                      ("Sinusoidal - Random", random_sinusoidal),
                                      ("He - Normal", kaiming_normal),
                                      ("He - Uniform", kaiming_uniform),
                                      ("Default", default_initialization),
                                      ("Orthogonal", orthogonal),
                                      ("LSUV", lsuv_with_dataloader)]:
        
        #for title, initialization in [("LSUV", lsuv_with_dataloader),]:
        
            print(title)
            # input_shape, num_classes = (300,), 300
            # input_shape, num_classes = (3, 32, 32), 100
            total_activations = None
            train_loader, val_loader, input_shape, num_classes = load_dataset(batch_size=32, size=224)
            # Create model and initialize weights 
            model = load_model(input_shape, num_classes)
            # Initialize
            if title == "LSUV":
                model = lsuv_with_dataloader(model, train_loader, device="cuda", verbose=False)
            else:
                model.apply(initialization)
            total_activations = None
            #for i in range(10):
            for i, (samples, labels) in enumerate(val_loader):
                if i == 30:
                    break
                samples = samples.normal_(mean=0, std=1)
                activations = get_activations(model, samples)
                if total_activations is None:
                    total_activations = activations
                    for layer in range(len(total_activations)):
                        total_activations[layer] = [activations[layer].detach().numpy()]
                else:
                    for layer in range(len(total_activations)):
                        total_activations[layer].append(activations[layer].detach().numpy())
    
            for layer in range(len(total_activations)):
                activations = total_activations[layer]
                activations = np.stack(activations, axis=0)
                activations = np.reshape(activations, shape=(-1, *activations.shape[3:]))
                activations = np.reshape(activations, shape=(activations.shape[0], -1))
                #activations = np.transpose(activations)
                total_activations[layer] = activations <= 0

            total_activations = total_activations[0]
            n_samples, m_neurons = total_activations.shape
            m_neurons = 250
            if samples_indexes is None:
                samples_indexes = np.random.choice(np.arange(0, np.min(total_activations.shape)), size=m_neurons, replace=False)
            total_activations = total_activations[samples_indexes][:,samples_indexes]
            plt.figure(figsize=(3, 2))
            cmap = plt.get_cmap('binary')  # 0 blanco, 1 negro
            #cmap = plt.get_cmap('viridis')
            tensor = total_activations

            img = plt.imshow(tensor, cmap=cmap, vmin=0, vmax=1,
                                        aspect="equal")
            
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f"Neurons")
            plt.ylabel("Samples")
            plt.title(title)

            plt.savefig(f'{plot_dir}activations_{title.lower()}.pdf', 
                        format='pdf', dpi=300,
                        bbox_inches='tight')
        plt.show()
        exit()

    if False: # OUI Values
        for title, initialization in [("Glorot - Normal", xavier_normal),
                                      ("Glorot - Uniform", xavier_uniform),
                                      ("Sinusoidal", fernandez_sinusoidal3),
                                      ("Sinusoidal - Random", random_sinusoidal),
                                      ("He - Normal", kaiming_normal),
                                      ("He - Uniform", kaiming_uniform),
                                      ("Default", default_initialization),
                                      ("Orthogonal", orthogonal),
                                      ("LSUV", lsuv_with_dataloader)]:
        
        #for title, initialization in [("LSUV", lsuv_with_dataloader),]:
        
            batch_size = 50
            # input_shape, num_classes = (300,), 300
            # input_shape, num_classes = (3, 32, 32), 100
            total_activations = None
            train_loader, val_loader, input_shape, num_classes = load_dataset(batch_size=32, size=224)
            # Create model and initialize weights 
            model = load_model(input_shape, num_classes)
            # Initialize
            if title == "LSUV":
                model = lsuv_with_dataloader(model, train_loader, device="cuda", verbose=False)
            else:
                model.apply(initialization)
            total_activations = None
            #for i in range(10):
            for i, (samples, labels) in enumerate(val_loader):
                if i == 30:
                    break
                #samples = samples.normal_(mean=0, std=1)
                activations = get_activations(model, samples)
                if total_activations is None:
                    total_activations = activations
                    for layer in range(len(total_activations)):
                        total_activations[layer] = [activations[layer].detach().numpy()]
                else:
                    for layer in range(len(total_activations)):
                        total_activations[layer].append(activations[layer].detach().numpy())
    
            for layer in range(len(total_activations)):
                activations = total_activations[layer]
                activations = np.stack(activations, axis=0)
                activations = np.reshape(activations, shape=(-1, *activations.shape[3:]))
                activations = np.reshape(activations, shape=(activations.shape[0], -1))
                #activations = np.transpose(activations)
                total_activations[layer] = activations <= 0

            total_activations = total_activations[0]
            n_samples, m_neurons = total_activations.shape
            m_neurons = 250
            # total_activations = total_activations[:m_neurons, :m_neurons]
            oui = on_batch_OUI(total_activations)
            print(f"{title}, {oui}", flush=True)
        exit()

    if False: # OUI plot
        initializations = ["He - Normal", "Orthogonal", "LSUV", "Sinusoidal"]
        oui_data = [np.array([0.9824, 0.7863, 0.6724, 0.6046, 0.5056, 0.4709]),
                    np.array([0.9897, 0.7991, 0.6623, 0.5907, 0.5452, 0.4944]),
                    np.array([0.9944, 0.7986, 0.6639, 0.5651, 0.5219, 0.4666]),
                    np.array([0.9864, 0.9824, 0.9861, 0.9824, 0.9805, 0.9824]),]
        
        vit_ouis_2ndlayer_1stmlp = '''
                    Glorot - Normal, 0.7060081845238094
                    Glorot - Uniform, 0.7183779761904762
                    Sinusoidal, 0.882905505952381
                    Sinusoidal - Random, 0.7564174107142856
                    He - Normal, 0.7721354166666667
                    He - Uniform, 0.8107328869047619
                    Default, 0.7678571428571429
                    Orthogonal, 0.8272879464285714
                    LSUV, 0.7775297619047619
                    '''
        vit_ouis_2ndlayer_11mlp = '''
                    Glorot - Normal, 0.2860863095238095
                    Glorot - Uniform, 0.2826450892857143
                    Sinusoidal, 0.853701636904762
                    Sinusoidal - Random, 0.5931919642857143
                    He - Normal, 0.5516183035714285
                    He - Uniform, 0.5855654761904762
                    Default, 0.6941964285714286
                    Orthogonal, 0.5423177083333333
                    LSUV, 0.6199776785714285
                    '''

    if False: # Test
        total_activations = None
        distribution = []
        for i in range(10):
        #for i, (samples, labels) in enumerate(train_loader):
            # if i == 2:
            #     break
            # shape = samples.shape
            samples = torch.empty(size=(batch_size, *input_shape))
            samples = samples.uniform_(-0.5, 0.5)
            #samples = samples.normal_(mean=0, std=1)
            activations = get_activations(model, samples)
            if total_activations is None:
                total_activations = activations
                distribution = [[samples.detach().numpy()]]
                for layer in range(len(total_activations)):
                    total_activations[layer] = [activations[layer].detach().numpy()]
            else:
                distribution[0].append(samples.detach().numpy())
                for layer in range(len(total_activations)):
                    total_activations[layer].append(activations[layer].detach().numpy())
 
        distribution[0] = np.stack(distribution[0], axis=0)
        #total_activations = total_activations[2:]
        layer_limits = [-0.5,]
        for layer in range(len(total_activations)):
            activations = total_activations[layer]
            activations = np.stack(activations, axis=0)
            activations = np.reshape(activations, shape=(-1, *activations.shape[2:]))
            activations = np.reshape(activations, shape=(activations.shape[0], -1))
            #activations = np.transpose(activations)
            distribution.append(activations)
            total_activations[layer] = activations <= 0
            print(activations.shape)
            layer_limits.append(activations.shape[-1] + layer_limits[-1])
      
        if True: # Distribution
            fig, axs = plt.subplots(
                1,                          # one row
                len(distribution),     # one column per layer
                sharex=False, sharey=False,
                gridspec_kw={
                    'wspace': 0.1,            # no horizontal padding
                    'hspace': 0             # (does nothing for a single row, but harmless)
                },
                figsize=(5, 3)    # optional: make canvas wide enough but keep height fixed
            )

            for i in range(len(axs)):
                tensor = distribution[i]

                img = axs[i].hist(tensor.flatten(), bins=100)
                
                #axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].set_xlabel(f"Layer {i}")
            
            #axs[1].set_title("Xavier normal intialization neuron activation state")
            axs[1].set_title("Kaiming normal")

        else: # QR-BARCODE
            widths = [layer.shape[-1] for layer in total_activations]

            fig, axs = plt.subplots(
                1,                          # one row
                len(total_activations),     # one column per layer
                sharex=False, sharey=False,
                gridspec_kw={
                    'width_ratios': widths, # <- goes here
                    'wspace': 0.1,            # no horizontal padding
                    'hspace': 0             # (does nothing for a single row, but harmless)
                },
                figsize=(5, 3)    # optional: make canvas wide enough but keep height fixed
            )

            cmap = plt.get_cmap('binary')  # 0 blanco, 1 negro
            #cmap = plt.get_cmap('viridis')
            for i in range(len(axs)):
                tensor = total_activations[i]

                img = axs[i].imshow(tensor, cmap=cmap, vmin=0, vmax=1,
                                            aspect="equal")
                
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].set_xlabel(f"Layer {i+1}")
            
            axs[0].set_ylabel("Samples")
            #axs[1].set_title("Xavier normal intialization neuron activation state")
            axs[1].set_title("Kaiming normal")

        # fig.savefig(f'{plot_dir}activations_{initialization.__name__}.pdf', 
        #             format='pdf', dpi=300,
        #             bbox_inches='tight')
        plt.show()
        exit()


if __name__ == '__main__':
    main()