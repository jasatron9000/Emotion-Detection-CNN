# MAIN
# =================================================================================================================
# The task of this file is to allow a user to use a custom dataset and train a available model easier by only
# inserting in the changeable parameters
# =================================================================================================================

# Imports needed for the following code to work
import torch
from image_processing import emotions
from training import trainer
from UI import userInput as user


# ===================================== Import Architectures =====================================
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

print(DEVICE)

# ===================================== Input user parameters =====================================
# print("Please select Architecture to use, enter a number to select the model: \n"
#       "1. AlexNet \n"
#       "2. SqeezeNet \n"
#       "3. VGG \n"
#       "4. ResNet \n"
#       "5. LeNet \n")
#
# # selected_model = input("Model: ")
# # while not selected_model.isdigit():
# #     print("Invalid string input, please enter an integer value")
# #     selected_model = input("Model: ")
# # selected_model = int(selected_model)
# # while not 1 <= selected_model <= len(models):
# #     print("Invalid input, please enter value between 1 and {}".format(len(models)))
# #     selected_model = int(input("Model: "))
# # selected_model = models[selected_model]
#
# if selected_model == RN:
#     print("Please select specific ResNet Model: \n"
#           "1. ResNet50 \n"
#           "2. ResNet101 \n"
#           "3. ResNet152 \n"
#           "4. ResNet34 \n"
#           "5. ResNet18 \n"
#           "6. ResNet110 \n")
#     selected_Resnet = input("Model: ")
#     while not selected_Resnet.isdigit():
#         print("Invalid string input, please enter an integer value")
#         selected_Resnet = input("Model: ")
#     selected_Resnet = int(selected_Resnet)
#     while not 1 <= selected_Resnet < len(ResNet):
#         print("Invalid input, please enter values between 1 and {}".format(len(ResNet)))
#         selected_Resnet = int(input("Model: "))
#     selected_ResNet = ResNet[selected_Resnet]
#     print(selected_ResNet)
#
#     print("Use for images size < 48 x 48 px?")
#     small = input("(y/n)")
#     while small != "y" and small != "n":
#         print("Invalid input, please enter either 'y' or 'n' ")
#         small = input("(y/n)")
#     net = selected_ResNet.to(DEVICE)
# else:
#     net = selected_model.to(DEVICE)
#
# print("Train model?")
# train = input("(y/n)")
# while train != "y" and train != "n":
#     print("Invalid input, please enter either 'y' or 'n' ")
#     train = input("(y/n)")
#
# Train = train == "y"
#
# print("Please input parameters: ")
# EPOCHS = int(input("Number of Epoch: "))
# BATCH_SIZE = int(input("Batch size: "))
# IMAGE_SIZE = int(input("Image size, please input a single integer as the image will be a square: "))
# TRAIN_PERCENT = float(input("TRAIN_PERCENT: "))
# lr = float(input("Learning rate: "))
#
# # Path locations on local device
# print("Please input the different paths needed to process images: ")
# DATA_LOCATION = str(input("Path to folder that contains all the sorted location folders: "))
# SAVE_LOCATION = str(input("Path to location to store the split data: "))
# MODEL_SAVE = str(input("Path to location where the trained model will be saved to: "))
# MODEL_NAME = str(input("Name of saved model: "))
#
# # Process the images from scratch
# print("Rebuild Data?")
# REBUILD_ANSWER = input("(y/n)")
# while REBUILD_ANSWER != "y" and REBUILD_ANSWER != "n":
#     print("Invalid input, please enter either 'y' or 'n' ")
#     REBUILD_ANSWER = input("(y/n)")
#
# REBUILD_DATA = REBUILD_ANSWER == "y"


uInt = user()
uInt.initValues()

# =========================================== Starts the code ===========================================

# Create the data and save
if uInt.DATA_REBUILD:
    rawData = emotions()
    rawData.make_training_data(uInt.DATA_LOCATION)
    rawData.save(uInt.SAVE_LOCATION, uInt.TRAIN_PERCENT)



uInt.SELECTED_MODEL = uInt.SELECTED_MODEL.to(DEVICE)

# Initialize the trainer with the necessary parameters
if uInt.FUNCTION == 1:
    trainBot = trainer(uInt.SELECTED_MODEL, DEVICE, uInt.SAVE_LOCATION,
                       uInt.EPOCHS, uInt.BATCH_SIZE, uInt.LR, uInt.MOMENTUM, uInt.WEIGHT_DECAY, uInt.FACTOR,
                       uInt.IMAGE_SIZE)
    trainBot.startTrain(uInt.MODEL_SAVE, uInt.MODEL_NAME, load=False)

elif uInt.FUNCTION == 2:
    trainBot = trainer(uInt.SELECTED_MODEL, DEVICE, uInt.SAVE_LOCATION,
                       uInt.EPOCHS, uInt.BATCH_SIZE, uInt.LR, uInt.MOMENTUM, uInt.WEIGHT_DECAY, uInt.FACTOR,
                       uInt.IMAGE_SIZE, evalMode=True)
    trainBot.loadCheckpoint(uInt.MODEL_SAVE, uInt.MODEL_NAME)
    trainBot.startTrain(uInt.MODEL_SAVE, uInt.MODEL_NAME, load=False)

elif uInt.FUNCTION == 3:
    trainBot = trainer(uInt.SELECTED_MODEL, DEVICE, uInt.SAVE_LOCATION,
                       uInt.EPOCHS, uInt.BATCH_SIZE, uInt.LR, uInt.MOMENTUM, uInt.WEIGHT_DECAY, uInt.FACTOR,
                       uInt.IMAGE_SIZE, evalMode=True)
    trainBot.loadCheckpoint(uInt.MODEL_SAVE, uInt.MODEL_NAME)
    trainBot.evaluateModel(uInt.MODEL_SAVE, uInt.MODEL_NAME)
else:
    print("Bye")
