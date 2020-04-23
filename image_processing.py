import os
import cv2

from tqdm import tqdm

import numpy as np

class emotions():
    training_data = []  # img of emotions

    #  a counter for each class to see how many images has been processed
    afraid_count = 0
    angry_count = 0
    disgust_count = 0
    happy_count = 0
    neutral_count = 0
    sad_count = 0
    surprised_count = 0

    def make_training_data(self, imgLocation_str, savedFilename_str, img_size_int):

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
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (img_size, img_size))
                    # print(img_size)
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
        np.save(savedFilename, self.training_data)
        print("Afraid:", self.afraid_count)
        print("Angry:", self.angry_count)
        print("Disgust:", self.disgust_count)
        print("Happy:", self.happy_count)
        print("Neutral:", self.neutral_count)
        print("Sad:", self.sad_count)
        print("Surprised:", self.surprised_count)
