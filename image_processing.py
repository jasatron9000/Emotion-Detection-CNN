import os
from PIL import Image as im
from tqdm import tqdm
import numpy as np
import torch.utils.data as data
import torchvision.transforms.functional as transforms


class emotions:
    training_data = []  # img of emotions

    #  a counter for each class to see how many images has been processed
    afraid_count = 0
    angry_count = 0
    disgust_count = 0
    happy_count = 0
    neutral_count = 0
    sad_count = 0
    surprised_count = 0

    def make_training_data(self, imgLocation: str, savedFilename: str, colour=True):

        # the location of the image files corresponding to class
        afraid = imgLocation + "/afraid"
        angry = imgLocation + "/angry"
        disgust = imgLocation + "/disgust"
        happy = imgLocation + "/happy"
        neutral = imgLocation + "/neutral"
        sad = imgLocation + "/sad"
        surprised = imgLocation + "/surprised"

        labels = {afraid: 0, angry: 1, disgust: 2, happy: 3, neutral: 4, sad: 5, surprised: 6}

        # iterates and processes through all the images from all the classes
        for emotion in labels:
            for f in tqdm(os.listdir(emotion)):
                try:
                    path = emotion + "/" + f
                    img = im.open(path)

                    # Inserts the photo
                    if not colour:
                        img = im.convert(mode="L")

                    self.training_data.append([np.array(img), labels[emotion]])

                    if emotion == afraid:
                        self.afraid_count += 1
                    elif emotion == angry:
                        self.angry_count += 1
                    elif emotion == disgust:
                        self.disgust_count += 1
                    elif emotion == happy:
                        self.happy_count += 1
                    elif emotion == neutral:
                        self.neutral_count += 1
                    elif emotion == sad:
                        self.sad_count += 1
                    elif emotion == surprised:
                        self.surprised_count += 1
                except Exception as e:
                    pass
                    # print(str(e))

        np.random.shuffle(self.training_data)
        #np.save(savedFilename, self.training_data)
        print("Afraid:", self.afraid_count)
        print("Angry:", self.angry_count)
        print("Disgust:", self.disgust_count)
        print("Happy:", self.happy_count)
        print("Neutral:", self.neutral_count)
        print("Sad:", self.sad_count)
        print("Surprised:", self.surprised_count)

    # Custom Transformations for our DataSet
    def Resize(self, size: int, crop=False):
        newTrainData = []

        for i in tqdm(range(len(self.training_data)), desc="Image Process: Resizing Images"):
            if crop:
                convertedImage = transforms.center_crop(self.training_data[i][0], size)
            else:
                convertedImage = transforms.resize(self.training_data[i][0], size)

            newTrainData.append([convertedImage, self.training_data[i][1]])

        print("Finished Processing: Array Length: " + str(len(newTrainData)))
        self.training_data = newTrainData

    # randomly crops an input image
    def randomCrop(img, width, height):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        cropped = img[y:y + height, x:x + width]
        return cropped

        # RandomCrop -> This method randomly crops every image 'amount' times and outputs them as a new

    # list of the flipped image
    def RandomCropData(self, amount: int, width: int, height: int):
        newTrainData = []

        # Iterates through the training data and appends 'amount' images which are cropped images with 'width' and 'height'
        # as the new dimensions of the images
        for i in tqdm(range(len(self.training_data)), desc="Image Process: Randomly Cropping All Images"):  # 4900
            crops = []
            count = amount
            while count > 0:
                convertedImage = randomCrop(self.training_data[i][0], width, height)

                if convertedImage not in crops:
                    crops.append([convertedImage, self.training_data[i][1]])
                else:
                    count += 1

                count -= 1
            # print(len(crops))
            for c in range(len(crops)):
                newTrainData.append(crops[c])

        print("Finished Processing: Array Length: " + str(len(newTrainData)))
        return newTrainData

    # Image Flip -> This method allows the user to get all the PIL images and flip them and outputs them as a new
    # list of the flipped image
    def ImageFlip(self):
        newTrainData = []

        # Iterates through the training data and appends two images which are the flipped and non-flipped image to
        # the new array
        for i in tqdm(range(len(self.training_data)), desc="Image Process: Flipping All Images"):
            convertedImage = transforms.hflip(self.training_data[i][0])

            newTrainData.append([convertedImage, self.training_data[i][1]])
            newTrainData.append([self.training_data[i][0]])

        print("Finished Processing: Array Length: " + str(len(newTrainData)))

        return newTrainData

    # TODO: Implement ChangeToTensor changes the list of PIL Images to a tensor

    # ChangeToTensor -> This method allows you to change the PIL image format to
    def ChangeToTensor(self, device: str):
        pass

class dataLoader(data.Dataset):
    def __init__(self, training_data):
        # Convert the training data of
        print(training_data[0][0].shape)
        self.image = training_data[0][0]
        print(training_data[0][1].shape)
        self.label = training_data[0][1]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return [self.image[index], self.label[index]]
