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
import VGG16 as VGG
import ResNet as RN
import LeNet as LN

models = {
    1: AN,
    2: SN,
    3: VGG,
    4: RN,
    5: LN
}

ResNet = {
    1: RN.ResNet50(),
    2: RN.ResNet101(),
    3: RN.ResNet152(),
    4: RN.ResNet34(),
    5: RN.ResNet18()
}

# ===================================== Input user parameters =====================================
print("Please select Architecture to use, enter a number to select the model: \n"
      "1. AlexNet \n"
      "2. SqeezeNet \n"
      "3. VGG \n"
      "4. ResNet \n"
      "5. LeNet \n")

selected_model = input("Model: ")
while not selected_model.isdigit():
    print("Invalid string input, please enter an integer value")
    selected_model = input("Model: ")
selected_model = int(selected_model)
while not 1 <= selected_model <= len(models):
    print("Invalid input, please enter value between 1 and {}".format(len(models)))
    selected_model = int(input("Model: "))
selected_model = models[selected_model]

if selected_model == RN:
    print("Please select specific ResNet Model: \n"
          "1. ResNet50 \n"
          "2. ResNet101 \n"
          "3. ResNet152 \n"
          "4. ResNet34 \n"
          "5. ResNet18 \n")
    selected_Resnet = input("Model: ")
    while not selected_Resnet.isdigit():
        print("Invalid string input, please enter an integer value")
        selected_Resnet = input("Model: ")
    selected_Resnet = int(selected_Resnet)
    while not 1 <= selected_Resnet < len(ResNet):
        print("Invalid input, please enter values between 1 and {}".format(len(ResNet)))
        selected_Resnet = int(input("Model: "))
    selected_Resnet = models[selected_Resnet]

    print("Use for images size < 48 x 48 px?")
    small = input("(y/n)")
    while small != "y" and small != "n":
        print("Invalid input, please enter either 'y' or 'n' ")
        small = input("(y/n)")
    net = selected_Resnet(small)
else:
    net = selected_model

print("Train model?")
Train = input("(y/n)")
while Train != "y" and Train != "n":
    print("Invalid input, please enter either 'y' or 'n' ")
    Train = input("(y/n)")

print("Please input parameters: ")
EPOCHS = int(input("Number of Epoch: "))
BATCH_SIZE = int(input("Batch size: "))
IMAGE_SIZE = int(input("Image size, please input a single integer as the image will be a square: "))
TRAIN_PERCENT = float(input("TRAIN_PERCENT: "))
lr = float(input("Learning rate: "))

# Path locations on local device
print("Please input the different paths needed to process images: ")
DATA_LOCATION = str(input("Path to folder that contains all the sorted location folders: "))
SAVE_LOCATION = str(input("Path to location to store the split data: "))
MODEL_SAVE = str(input("Path to location where the trained model will be saved to: "))
MODEL_NAME = str(input("Name of saved model: "))

# Process the images from scratch
print("Rebuild Data?")
REBUILD_DATA = input("(y/n)")
while REBUILD_DATA != "y" and REBUILD_DATA != "n":
    print("Invalid input, please enter either 'y' or 'n' ")
    REBUILD_DATA = input("(y/n)")

# =========================================== Starts the code ===========================================

# Initialising the device to be used for making the CNN calculations
DEVICE = None
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
