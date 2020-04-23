from image_process import emotions
from trainAlgorithm import trainer
import sys

sys.path.insert(1, 'models')

#Constants
REBUILD_DATA = False
IMAGE_SIZE = 0
EPOCHS = 0
BATCH_SIZE = 0
folderName = ""
fileName = "AlexNet-Data"


#Save and Load Data



#Augment the Data to suite the
training_data = ip.FlipImage(training_data)
training_data = ip.RandCrop(training_data, 5, (227,227))
training_data = ip.ConvertToTensor(training_data)

#Training the Data
train = trainer()

train.start(EPOCHS, BATCH_SIZE)
train.result()
train.saveResult("AlexNet-Results")