import torch
from image_processing import emotions
from training import trainer
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import time
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

import sys

sys.path.insert(1, 'models')

import AlexNet as AN
import SqueezeNet as SN

# Constants
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = 50
CROP_SIZE = 50
REBUILD_DATA = False
DEVICE = None
TRAIN_PERCENT = 0.7
DATA_LOCATION = "D:/Biggie Cheese/Desktop/Uni/302/Data/KDEF Updated"  # FILE LOCATION OF THE DATA
SAVE_LOCATION = "D:/Biggie Cheese/Desktop/a"  # WHERE YOU WANT TO SAVE THE AUGMENTED DATA
LOAD_LOCATION = "D:/Biggie Cheese/Desktop/a"

# Initialising the device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    DEVICE_STATUS = True
    print("\nUsing the GPU")
else:
    DEVICE = torch.device("cpu")
    DEVICE_STATUS = False
    print("\nUsing the CPU")

# Retrieve and Augment Data
if REBUILD_DATA:
    rawData = emotions(IMAGE_SIZE)
    rawData.make_training_data(DATA_LOCATION)
    # rawData.ImageFlip()
    # rawData.RandomCropData(5, CROP_SIZE, CROP_SIZE)
    rawData.save(SAVE_LOCATION, TRAIN_PERCENT)
    LOAD_LOCATION = SAVE_LOCATION

train = ImageFolder(LOAD_LOCATION + "/train", transform=transforms.Compose([transforms.ToTensor()]))
valid = ImageFolder(LOAD_LOCATION + "/validate", transform=transforms.Compose([transforms.ToTensor()]))
test = ImageFolder(LOAD_LOCATION + "/test", transform=transforms.Compose([transforms.ToTensor()]))
print("\nIMAGES HAS BEEN RETRIEVED")

trainSet = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
validSet = data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)
testSet = data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
print("\nIMAGES HAS BEEN LOADED IN THE PROGRAM")

net = AN.TestClass(CROP_SIZE).to(DEVICE)

trainBot = trainer(net, trainSet, validSet, testSet)

trainBot.startTrain(EPOCHS, DEVICE, BATCH_SIZE)
