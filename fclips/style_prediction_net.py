import torch.nn as nn

class TextStylePredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 150)
        self.fc4 = nn.Linear(150, 150)
        self.fc5 = nn.Linear(150, 100)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.tanh(self.fc5(x))
        return x
