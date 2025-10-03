import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from common import *
from initialize import *
from my_datasets import *
from models import *
from lsuv import lsuv_with_dataloader

EPOCHS = 100
LR = 0.001

def main():
    # dataset_name = cifar100 tiny-imagenet tiny-imagenet imagenet1k wikitext
    # model_name =   resnet50 mobilenet     efficientnet  vit        roberta
    # for optimizer in [optim.SGD, optim.Adam, optim.AdamW]:
    # for initialization_name in [fernandez_sinusoidal3, default_initialization, orthogonal, lsuv_with_dataloader]:
    dataset_name = "cifar100"
    model_name = "resnet50"
    optimizer = optim.AdamW
    initialization = fernandez_sinusoidal_random
    #initialization = zero
    
    if dataset_name == "cifar100": 
        dataset = load_cifar100
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
    train_loader, test_loader, input_shape, num_classes = dataset(size=224)

    # Create model and initialize weights 
    model = model(input_shape, num_classes)
    # Initialize
    if initialization.__name__ == "lsuv_with_dataloader":
        model = lsuv_with_dataloader(model, train_loader)
    else:
        model.apply(initialization)  # Apply custom initialization

    # Train
    #model = torch.compile(model)
    model = Model(predefined_model=model)
    model.compile(
        loss=nn.CrossEntropyLoss(), 
        optimizer=optimizer, 
        optimizer_params = {"lr": 0.001, "weight_decay": 0.001},
        metrics=Accuracy())
    #model.add_callback([EarlyStopping(patience=50, delta=0)])
    history = model.train_model(train_loader, test_loader, num_epochs=EPOCHS, verbose=True)
    print(history)


if __name__ == '__main__':
    main()