import torch
import torch.nn as nn
import torch.optim as optim

from common import *
from initialize import *
from my_datasets import *
from my_models import *

EPOCHS = 2
LR = 0.001

def main():
    dataset = load_cifar100
    model = load_mobilenet

    # Load data
    train_loader, test_loader, input_shape, num_classes = dataset()

    # Create model and initialize weights 
    model = model(input_shape, num_classes)
    model.apply(sinusoidal_init)

    # Train
    model = Model(predefined_model=model)
    model.compile(
        loss=nn.CrossEntropyLoss(), 
        optimizer=optim.AdamW, 
        optimizer_params = {"lr": 0.001, "weight_decay": 0.001},
        metrics=Accuracy())
    history = model.train_model(train_loader, test_loader, num_epochs=EPOCHS, verbose=True)
    print(history)

if __name__ == '__main__':
    main()