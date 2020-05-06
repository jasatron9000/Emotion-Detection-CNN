import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # Input size = 64
    def __init__(self):
        super().__init__()

        # Defining convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=2)

        # Defining fully connected layers
        # The last fc layer will have 7 as output for our case as there are 7 emotions
        self.fc1 = nn.Linear(12800, 500)
        self.fc2 = nn.Linear(500, 7)

        # Defining pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):

        # Go through the convolution layers (activation function included)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)  # Pooling
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)# Pooling

        x = x.reshape(x.shape[0], -1)  # flatten

        # pass through the FC layers (activation function included)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x



