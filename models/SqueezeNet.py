import torch
import torch.nn as nn
import torch.nn.functional as F


class Fire(nn.Module):
    def __init__(self, input, squeeze, expand3x3, expand1x1):
        super().__init__()

        # Define Squeeze Layer
        self.squeezeConv = nn.Conv2d(input, squeeze, 1)

        # Define Expand Layer
        self.expandConv3x3 = nn.Conv2d(squeeze, expand3x3, 3, padding=1)
        self.expandConv1x1 = nn.Conv2d(squeeze, expand1x1, 1)

    def forward(self, x):
        x = F.relu(self.squeezeConv(x), inplace=True)
        expand1x1 = F.relu(self.expandConv1x1(x), inplace=True)
        expand3x3 = F.relu(self.expandConv3x3(x), inplace=True)
        return torch.cat([expand1x1, expand3x3], 1)


class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, 7, 2)
        self.fire1 = Fire(96, 16, 64, 64)
        self.fire2 = Fire(128, 16, 64, 64)
        self.fire3 = Fire(128, 32, 128, 128)
        self.fire4 = Fire(256, 32, 128, 128)
        self.fire5 = Fire(256, 48, 192, 192)
        self.fire6 = Fire(384, 48, 192, 192)
        self.fire7 = Fire(384, 64, 256, 256)
        self.fire8 = Fire(512, 64, 256, 256)
        self.conv2 = nn.Conv2d(512, 7, kernel_size=1, stride=1)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=3, stride=2)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.fire8(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.softmax(x, dim=0).view(-1, 7)
        return x
