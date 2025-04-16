import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from common import *
from initialize import *
from datasets.cifar import load_cifar100
from models.lenet import lenet5
from models.resnet50 import resnet50

EPOCHS = 100
LR = 0.001

def main():
    sys.stdout = Logger("./results/output.log")

    initialization_name = "fernandez"
    dataset_name = "cifar100"
    model_name = "resnet50"

    for initialization_name in ["fernandez", "default"]:
        for optimizer in [optim.SGD, optim.Adam, optim.AdamW]:
            if initialization_name == "fernandez":
                initialization = fernandez_initialization
            else:
                initialization = default_initialization        
            if dataset_name == "cifar100": 
                dataset = load_cifar100
            if model_name == "resnet50":
                model = resnet50
            elif model_name == "lenet5":
                model = lenet5


            print(f"{model_name}; {dataset_name}; {initialization_name}; {optimizer.__name__}", flush=True)
            # Load data
            train_loader, test_loader, input_shape, num_classes = dataset(size=256)

            for iteration in range(1):
                # Create model and initialize weights 
                model = resnet50(input_shape, num_classes)
                # Initialize
                model.apply(initialization)  # Apply custom initialization

                # Train
                model = Model(predefined_model=model)
                model.compile(
                    loss=nn.CrossEntropyLoss(), 
                    optimizer=optimizer, 
                    optimizer_params = {"lr": 0.001, "weight_decay": 0.001},
                    metrics=Accuracy())
                model.add_callback([{"type": "early_stopping", "patience": 10, "delta": 0.005}])
                history = model.train_model(train_loader, test_loader, num_epochs=EPOCHS, verbose=True)
                print(history)


if __name__ == '__main__':
    main()