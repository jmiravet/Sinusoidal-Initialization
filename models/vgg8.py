import torch
import torch.nn as nn

class VGG8(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG8, self).__init__()
        self.feature_extraction = nn.Sequential(
            # Conv 1
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            # Conv 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            # Conv 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            # Conv 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            # Conv 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten())
                      
        # Compute the flatten output shape for the classification module
        with torch.no_grad():
            random_input = torch.rand(size=(1, *input_shape))
            output = self.feature_extraction(random_input)
            output_shape = output.shape[-1]
        
        self.classification = nn.Sequential(
            nn.Linear(output_shape, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        x = self.feature_extraction(x)
        out = self.classification(x)
        return out