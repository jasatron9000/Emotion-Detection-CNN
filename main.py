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

# Retrieve and Augment Data
data = emotions()

if REBUILD_DATA:
    # Retrieve Data
    pass
    # Augment Data

else:
    # Load Tensors
    pass

# Insert it in the DataLoader

# Create the model and train/validate it
