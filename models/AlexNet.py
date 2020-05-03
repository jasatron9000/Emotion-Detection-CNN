import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, size):
        super().__init__()

        # Define Convolution Layers
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)

        # Finding out the total nodes after the convolution layers
        x = torch.randn(3, size, size).view(-1, 3, size, size)
        self.to_linear = -1
        self.calcConvToLinear(x)

        # Define Fully-Connected Layers
        self.fc1 = nn.Linear(self.to_linear, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 7)

        # Define Dropout Regularisation
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    # Putting the convolution layers through its Pooling and Non-Linearity
    def calcConvToLinear(self, x):
        # Pooling and Non-Linearity
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 3), stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 3), stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), (3, 3), stride=2)

        # Check if the total node has been calculated
        if self.to_linear is -1:
            self.to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    # Behaviour of the Network
    def forward(self, x):
        x = self.calcConvToLinear(x)
        x = x.view(-1, self.to_linear)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))

        x = self.dropout1(x)
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return F.softmax(x, dim=1)


class TestClass(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.classify = nn.Sequential(
            nn.Linear(31*31*2, 7),
            nn.Softmax(dim=1)
        )

        # Behaviour of the Network

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 31*31*2)
        x = self.classify(x)

        return x
