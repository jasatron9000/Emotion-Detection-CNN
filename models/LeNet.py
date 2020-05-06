import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Defining convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=2)

        # Defining fully connected layers
        # The last fc layer will have 7 as output for our case as there are 7 emotions
        self.fc1 = nn.Linear(3200, 500)
        self.fc2 = nn.Linear(500, 7)

        # Defining pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):

        # ========================================= Detect input errors ================================================
        input_size = 32
        num_channel = 1
        if x.shape[1] != num_channel or x.shape[2] != input_size or x.shape[3] != input_size:
            raise Exception("Input image does not have correct dimensions, "
                            "please check [IMAGE_SIZE] is {} and image is in grayscale".format(input_size))
        # ==============================================================================================================

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


