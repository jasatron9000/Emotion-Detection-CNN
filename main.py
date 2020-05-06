# MAIN
# =================================================================================================================
# The task of this file is to allow a user to use a custom dataset and train a available model easier by only
# inserting in the changeable parameters
# =================================================================================================================

# Imports needed for the following code to work
import torch
from image_processing import emotions
from training import trainer
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import sys
sys.path.insert(1, 'models')


# ===================================== Import Architectures =====================================

# Imports needed for the following code to work
import AlexNet as AN
import SqueezeNet as SN
import VGG16
import ResNet3418 as RN


# ===================================== Input user parameters =====================================

# Constants
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = 32
TRAIN_PERCENT = 0.7
lr = 0.001

# Process the images from scratch
REBUILD_DATA = False
DEVICE = None

# Path locations on local device
DATA_LOCATION = r"D:\2020\COMPSYS 302\picturs\original\sorted_emotion"  # FILE LOCATION OF THE DATA
SAVE_LOCATION = r"D:\2020\COMPSYS 302\CNNs"
MODEL_SAVE = r"D:\2020\COMPSYS 302\CNNs"
MODEL_NAME = "ResNet18_ADAM_LR_0.001_64x64 DROPOUT "

# Network selection + to train or not
net = RN.ResNet18().to(DEVICE)
Train = True


# =========================================== Starts the code ===========================================

# Initialising the device to be used for making the CNN calculations
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    DEVICE_STATUS = True
    print("\nUsing the GPU")
else:
    DEVICE = torch.device("cpu")
    DEVICE_STATUS = False
    print("\nUsing the CPU")

# Create the data and save
if REBUILD_DATA:
    rawData = emotions()
    rawData.make_training_data(DATA_LOCATION)
    rawData.save(SAVE_LOCATION, TRAIN_PERCENT)

# Image augmentation is applied to the processed images
transformAugmented = transforms.Compose([transforms.Resize(int(IMAGE_SIZE*1.1)),
                                         transforms.RandomCrop(IMAGE_SIZE),
                                         transforms.Grayscale(1),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomAffine(10),
                                         transforms.ToTensor()])

transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.RandomAffine(10),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])

# Initialize the datasets used for developing the CNN
train = ImageFolder(SAVE_LOCATION + "\\train", transform=transformAugmented)
valid = ImageFolder(SAVE_LOCATION + "\\validate", transform=transformAugmented)
test = ImageFolder(SAVE_LOCATION + "\\test", transform=transformAugmented)
print("\nIMAGES HAS BEEN RETRIEVED")

# Initialize initial weights for network
classWeights = torch.zeros((1, 7))
for _, label in train:
    classWeights[0][label] += 1

for _, label in valid:
    classWeights[0][label] += 1

for _, label in test:
    classWeights[0][label] += 1

classWeights = 1/classWeights
classWeights = classWeights.to(DEVICE)
print(classWeights)

# Load the processed images that are ready for calculation into the program
trainSet = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
validSet = data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)
testSet = data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
print("\nIMAGES HAS BEEN LOADED IN THE PROGRAM")

# Initialize the trainer with the necessary parameters
trainBot = trainer(EPOCHS, BATCH_SIZE, net, trainSet, validSet, testSet, DEVICE, lr, weights=classWeights)
if Train:
    trainBot.startTrain(MODEL_SAVE, MODEL_NAME, load=False)
else:
    trainBot.loadCheckpoint(MODEL_SAVE, MODEL_NAME)
    trainBot.evaluateModel(MODEL_SAVE, MODEL_NAME)
