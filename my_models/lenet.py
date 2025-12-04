import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LeNet5, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_shape[0], 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten())
        
        # Compute the flatten output shape for the classification module
        with torch.no_grad():
            random_input = torch.rand(size=(1, *input_shape))
            output = self.feature_extraction(random_input)
            output_shape = output.shape[-1]
        
        self.classification = nn.Sequential(
            nn.Linear(output_shape, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes))
        
    def forward(self, x):
        x = self.feature_extraction(x)
        out = self.classification(x)
        return out
    
def load_lenet5(input_shape, num_classes):
    model = LeNet5(input_shape=input_shape, num_classes=num_classes)
    return model
