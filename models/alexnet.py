import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            # Conv 1
            nn.Conv2d(input_shape[0], 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv 2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv 3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(384),
            # Conv 4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(384),
            # Conv 5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten())
              
        # Compute the flatten output shape for the classification module
        with torch.no_grad():
            random_input = torch.rand(size=(1, *input_shape))
            output = self.feature_extraction(random_input)
            print(output.shape)
            output_shape = output.shape[-1]
        
        self.classification = nn.Sequential(
            nn.Linear(output_shape, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes))
        

    def forward(self, x):
        x = self.feature_extraction(x)
        out = self.classification(x)
        return out

