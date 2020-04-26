import torch
from image_processing import emotions
from training import trainer, loadTensors
import torch.utils.data as data
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

import sys

sys.path.insert(1, 'models')

import AlexNet as AN

# Constants
EPOCHS = 1
BATCH_SIZE = 1
IMAGE_SIZE = 256
REBUILD_DATA = True
DEVICE = None
TRAIN_PERCENT = 0.7
DATA_LOCATION = "D:/Biggie Cheese/Desktop/Uni/302/Data/KDEF Updated" # FILE LOCATION OF THE DATA
SAVE_LOCATION = "D:\\Biggie Cheese\\Desktop\\a" # WHERE YOU WANT TO SAVE THE AUGMENTED DATA

# Initialising the device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Using the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using the CPU")

# Retrieve and Augment Data
rawData = emotions(IMAGE_SIZE)
rawData.make_training_data(DATA_LOCATION)
rawData.ImageFlip()
rawData.RandomCropData(5, 227, 227)
rawData.save(SAVE_LOCATION, TRAIN_PERCENT)
