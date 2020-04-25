import torch
from image_processing import dataLoader, emotions
from training import trainer

import sys

sys.path.insert(1, 'models')

from AlexNet import AlexNet

# Constants
EPOCHS = 1
BATCH_SIZE = 1
IMAGE_SIZE = 1
REBUILD_DATA = True
DEVICE = None

print("a")

# Initialising the device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Using the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using the CPU")


# Retrieve and Augment Data
data = emotions()

if REBUILD_DATA:
    # Retrieve Data
    data.make_training_data("D:/Biggie Cheese/Desktop/Uni/302/Data/KDEF Updated", "training.npy")

    # Augment Data
    data.Resize(227)
    data.ImageFlip()

    data.save("AlexNet-Images", "AlexNet-Labels")
else:
    # Load Tensors
    pass

# Insert it in the DataLoader
net = AlexNet()

# Create the model and train/validate it
train = trainer(net, )