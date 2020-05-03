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
import VGG16
import ResNet as RN

# Constants
EPOCHS = 50
BATCH_SIZE = 64
IMAGE_SIZE = 64
REBUILD_DATA = False
DEVICE = None
TRAIN_PERCENT = 0.7
DATA_LOCATION = r"D:\Biggie Cheese\Desktop\Uni\302\CS302-Python-2020-Group25\data"  # FILE LOCATION OF THE DATA
SAVE_LOCATION = r"D:\Biggie Cheese\Desktop\Uni\302\CS302-Python-2020-Group25\edited"  #
LOAD_LOCATION = r"D:\Biggie Cheese\Desktop\Uni\302\CS302-Python-2020-Group25\edited"
MODEL_SAVE = r"D:\Biggie Cheese\Desktop\Uni\302\CS302-Python-2020-Group25\Results\Edited FER\VGG\VGG13_ADAM_LR_0.001-TRANSFORMED-LESS"
MODEL_NAME = "VGG13_ADAM_LR_0.001-TRANSFORMED-LESS "

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

transformAugmented = transforms.Compose([transforms.Resize(int(IMAGE_SIZE*1.1)),
                                         transforms.RandomCrop(IMAGE_SIZE),
                                         transforms.Grayscale(1),
                                         transforms.RandomVerticalFlip(),
                                         transforms.ToTensor()])

transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.RandomAffine(10),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor()])

train = ImageFolder(LOAD_LOCATION + "\\train", transform=transformAugmented)
valid = ImageFolder(LOAD_LOCATION + "\\validate", transform=transformAugmented)
test = ImageFolder(LOAD_LOCATION + "\\test", transform=transformAugmented)
print("\nIMAGES HAS BEEN RETRIEVED")

classWeights = torch.zeros((1, 7))

for _, label in train:
    classWeights[0][label] += 1

for _, label in valid:
    classWeights[0][label] += 1

for _, label in test:
    classWeights[0][label] += 1

classWeights = 1 / classWeights
classWeights = classWeights.to(DEVICE)

print(classWeights)

trainSet = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
validSet = data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)
testSet = data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
print("\nIMAGES HAS BEEN LOADED IN THE PROGRAM")

net = VGG16.CustomVGG13().to(DEVICE)

trainBot = trainer(EPOCHS, BATCH_SIZE, net, trainSet, validSet, testSet, DEVICE, lr=0.001, weights=classWeights)

trainBot.startTrain(MODEL_SAVE, MODEL_NAME, load=False)

# trainBot.loadCheckpoint(MODEL_SAVE, MODEL_NAME)
# trainBot.evaluateModel(MODEL_SAVE, MODEL_NAME)
