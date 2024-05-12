import torch.nn as nn

class ParameterPrediction(nn.Module):
    def __init__(self, bottleneck=512):
        super().__init__()
        self.fc1 = nn.Linear(512, bottleneck)
        self.fc2 = nn.Linear(bottleneck, bottleneck)
        self.fc3 = nn.Linear(bottleneck, bottleneck)
        self.fc4 = nn.Linear(bottleneck, 512)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

