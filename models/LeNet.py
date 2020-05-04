import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Defining convolution layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Defining fully connected layers
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)

        # The last fc layer will have 7 as output for our case as there are 7 emotions
        self.fc3 = nn.Linear(84, 7)

        # Defining pooling layer
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Go through the convolution layers (activation function included)
        x = torch.tanh(self.conv1(x))
        x = self.avgpool(x)  # Pooling
        x = torch.tanh(self.conv2(x))
        x = self.avgpool(x)# Pooling

        x = x.reshape(x.shape[0], -1)  # flatten
        print(x.shape)

        # pass through the FC layers (activation function included)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
