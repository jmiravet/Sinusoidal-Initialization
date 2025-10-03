import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, json

from common import *
from initialize import *
from my_datasets import *
from models import *
from lsuv import lsuv_with_dataloader

EPOCHS = 100
LR = 0.001

def main(initialization_name):
    # dataset_name = cifar100 tiny-imagenet tiny-imagenet imagenet1k wikitext
    # model_name =   resnet50 mobilenet     efficientnet  vit        roberta
    # for optimizer in [optim.SGD, optim.Adam, optim.AdamW]:
    # for initialization_name in ["fernandez", "default"]:
    dataset_name = "cifar100"
    model_name = "resnet50"
    optimizer = optim.SGD
    
    output_file = f"./results/output{dataset_name}-{model_name}-{initialization_name}-{optimizer.__name__}.log"
    sys.stdout = Logger(output_file)
    print(f"{model_name}; {dataset_name}; {initialization_name}; {optimizer.__name__}", flush=True)

    if initialization_name == "default":
        initialization = default_initialization
    elif initialization_name == "fernandez":
        initialization = fernandez_sinusoidal3
    elif initialization_name == "ortogonal":
        initialization = init_orthogonal
    elif initialization_name == "lsuv":
        initialization = lsuv_with_dataloader
    elif initialization_name == "zero":
        initialization = zero_init
    elif initialization_name == "blumenfeld":
        initialization = blumenfeld_init
    elif initialization_name == "arcsinerandom":
        initialization = arcsine_random_init
    else:
        raise ValueError("Unknown initialization")
    
    if dataset_name == "cifar100": 
        dataset = load_cifar100
    elif dataset_name == "cifar10":
        dataset = cifar.load_cifar10
    elif dataset_name == "tiny-imagenet":
        dataset = load_tinyimagenet
    elif dataset_name == "imagenet1k":
        dataset = load_imagenet1k
    if model_name == "resnet50":
        model = load_resnet50
    elif model_name == "mobilenet":
        model = load_mobilenet
    elif model_name == "efficientnet":
        model = load_efficientnet
    elif model_name == "vit":
        model = load_vit
    elif model_name == "lenet5":
        model = load_lenet5

    output_file = f"./results/output{dataset_name}-{model_name}-{initialization.__name__}-{optimizer.__name__}.log"
    sys.stdout = Logger(output_file)
    print(f"{model_name}; {dataset_name}; {initialization.__name__}; {optimizer.__name__}", flush=True)

    # Load data
    train_loader, test_loader, input_shape, num_classes = dataset()

    # Create model and initialize weights 
    model = model(input_shape, num_classes)
    # Initialize
    if initialization_name == "lsuv":
        model = lsuv_with_dataloader(model, train_loader) 
    else:
        model.apply(initialization)

    # Train
    model = Model(predefined_model=model)
    model.compile(
        loss=nn.CrossEntropyLoss(), 
        optimizer=optimizer, 
        optimizer_params = {"lr": 0.001, "weight_decay": 0.001},
        metrics=Accuracy())
    #model.add_callback([EarlyStopping(patience=50, delta=0)])
    history = model.train_model(train_loader, test_loader, num_epochs=EPOCHS, verbose=True)
    print(history)
    log_path = os.path.join(os.path.dirname(__file__), f"{dataset_name}_{model_name}_{initialization_name}.log")
    with open(log_path, "w") as f: json.dump(history, f, indent=2)

if __name__ == '__main__':
    for initialization_name in ["fernandez"]: # ["default", "fernandez",  "ortogonal", "lsuv", "zero", "blumenfeld", "arcsinerandom"]:
        main(initialization_name)