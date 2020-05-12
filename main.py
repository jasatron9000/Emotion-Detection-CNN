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
