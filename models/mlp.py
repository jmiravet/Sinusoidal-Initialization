import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MLP, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Flatten())
        
        # Compute the flatten output shape for the classification module
        with torch.no_grad():
            random_input = torch.rand(size=(1, *input_shape))
            output = self.feature_extraction(random_input)
            output_shape = output.shape[-1]
        
        self.classification = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            #nn.BatchNorm1d(300),
            nn.Linear(300, 300),
            nn.ReLU(),
            #nn.BatchNorm1d(300),
            nn.Linear(300, 300),
            nn.ReLU(),
            #nn.BatchNorm1d(300),
            nn.Linear(300, 300),
            nn.ReLU(),
            #nn.BatchNorm1d(300),
            nn.Linear(300, 300),
            nn.ReLU(),
            #nn.BatchNorm1d(300),
            nn.Linear(300, 300))
        
    def forward(self, x):
        x = self.feature_extraction(x)
        out = self.classification(x)
        return out
    
def load_mlp(input_shape, num_classes):
    model = MLP(input_shape=input_shape, num_classes=num_classes)
    return model