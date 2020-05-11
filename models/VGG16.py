import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomVGG13(nn.Module):
    def __init__(self):
        super().__init__()

        # A pytorch Sequential that goes through all the convolution layers in the network responsible for picking up
        # the features of the image.
        #
        # Convolution block layer structure: 2 -> 2 -> 3 -> 3
        #
        # Each convolution layer is paired with a batch normalisation and ReLu activation function
        #
        # Each block of convolution ends with a pooling layer which halves the map dimension
        self.features = nn.Sequential(

            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # A pytorch Sequential that goes through all the fully connected layers(3)
        #
        # Each fully connected layer is paired with a batch normalisation and ReLu activation function
        #
        # The dimensions has been halved 4 times the input size of 64 is reduced to 4 x 4 and the last
        # convolution layer has 256
        self.classify = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(4*4*256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 7),
            nn.BatchNorm1d(7),
            nn.ReLU()

        )

    def forward(self, x):

        # ========================================= Detect input errors ================================================
        input_size = 64
        num_channel = 1
        if x.shape[1] != num_channel or x.shape[2] != input_size or x.shape[3] != input_size:
            raise Exception("Input image does not have correct dimensions, "
                            "please check [IMAGE_SIZE] is {} and image is in grayscale".format(input_size))
        # ==============================================================================================================

        # Go through convolution layers
        x = self.features(x)

        x = x.view(-1, 4 * 4 * 256)  # flatten

        # Go through fully connected layers
        x = self.classify(x)

        return x

class CustomVGG13x96(nn.Module):
    # Implementation is same for the 96 x 96 input apart from the classify section where the dimensions
    # are different due to the increased input dimensions
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # SAME AS CustomVGG13
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classify = nn.Sequential(

            # A pytorch Sequential that goes through all the fully connected layers(3)
            #
            # Each fully connected layer is paired with a batch normalisation and ReLu activation function
            #
            # The dimensions has been halved 4 times the input size of 96 is reduced to 6 x 6 and the last
            # convolution layer has 256
            #
            # The increase in linear output is also increased to compensate for the increased input size
            nn.Dropout(p=0.25),
            nn.Linear(6*6*256, 2304),
            nn.BatchNorm1d(2304),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2304, 2304),
            nn.BatchNorm1d(2304),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2304, 7),
            nn.BatchNorm1d(7),
            nn.ReLU(),
            nn.Dropout(p=0.5),

        )

    def forward(self, x):

        # ========================================= Detect input errors ================================================
        input_size = 96
        num_channel = 1
        if x.shape[1] != num_channel or x.shape[2] != input_size or x.shape[3] != input_size:
            raise Exception("Input image does not have correct dimensions, "
                            "please check [IMAGE_SIZE] is {} and image is in grayscale".format(input_size))
        # ==============================================================================================================

        # Go through convolution layers
        x = self.features(x)

        x = x.reshape(x.shape[0], -1)  # Flatten

        # Go through fully connected layers
        x = self.classify(x)

        return x
