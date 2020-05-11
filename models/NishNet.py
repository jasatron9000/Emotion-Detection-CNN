import torch
import torch.nn as nn
import torch.nn.functional as F

output_features = 64
conv_dropout_p = 0.5
fc_dropout_p = 0.4
num_classes = 7

class NishNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Defining the convolution and batch normalization layers
        #
        # There are 8 convolution layers and 4 batch normalisation layer in total
        self.conv1 = nn.Conv2d(1, output_features, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(output_features, output_features, 3, padding=(1, 1))
        self.BN1 = nn.BatchNorm2d(output_features)

        self.conv3 = nn.Conv2d(output_features, output_features * 2, 3, padding=(1, 1))
        self.BN2 = nn.BatchNorm2d(output_features * 2)
        self.conv4 = nn.Conv2d(output_features * 2, output_features * 2, 3, padding=(1, 1))

        self.conv5 = nn.Conv2d(output_features * 2, output_features * 2 * 2, 3, padding=(1, 1))
        self.BN3 = nn.BatchNorm2d(output_features * 2 * 2)
        self.conv6 = nn.Conv2d(output_features * 2 * 2, output_features * 2 * 2, 3, padding=(1, 1))

        self.conv7 = nn.Conv2d(output_features * 2 * 2, output_features * 2 * 2 * 2, 3, padding=(1, 1))
        self.BN4 = nn.BatchNorm2d(output_features * 2 * 2 * 2)
        self.conv8 = nn.Conv2d(output_features * 2 * 2 * 2, output_features * 2 * 2 * 2, 3, padding=(1, 1))

        # Define 2 python dictionaries to store the layers for cleaner implementation later on
        self.conv_layers = {
            1: self.conv3,
            2: self.conv4,
            3: self.conv5,
            4: self.conv6,
            5: self.conv7,
            6: self.conv8
        }
        self.norm_layers = {
            1: self.BN2,
            3: self.BN3,
            5: self.BN4
        }

        # Define fully connected layers
        self.fc1 = nn.Linear(8192, output_features * 2 * 2 * 2)
        self.fc2 = nn.Linear(output_features * 2 * 2 * 2, output_features * 2 * 2)
        self.fc3 = nn.Linear(output_features * 2 * 2, output_features * 2)
        self.fc4 = nn.Linear(output_features * 2, num_classes)


    def convs(self, x, conv_layer: int, norm_layer: int):
        x = F.relu(self.conv_layers[conv_layer](x))
        x = self.norm_layers[norm_layer](x)
        x = F.relu(self.conv_layers[conv_layer + 1](x))
        x = self.norm_layers[norm_layer](x)
        x = F.dropout(F.max_pool2d(x, (2, 2), 2), p=conv_dropout_p, training=True)

        return x

    def forward(self, x):
        # pass through the conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.BN1(x)
        x = F.dropout(F.max_pool2d(x, (2, 2), 2), p=conv_dropout_p, training=True)

        # Iterate through the repeating convolution layers
        for i in [1, 3, 5]:
            x = self.convs(x, i, i)

        # flatten
        x = x.reshape(x.shape[0], -1)

        # pass through the FC layers
        x = F.dropout(F.relu(self.fc1(x)), p=fc_dropout_p, training=True)
        x = F.dropout(F.relu(self.fc2(x)), p=fc_dropout_p, training=True)
        x = F.dropout(F.relu(self.fc3(x)), p=fc_dropout_p, training=True)
        x = self.fc4(x)

        return x

def test():
    net = NishNet()
    y = net(torch.randn(4, 1, 64, 64))
    print(y.size())

test()