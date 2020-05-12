# CS302-Python-2020-Group25

## Introduction
This repository contains the nessasary python files for training and and saving Convolutional Neural Networks(CNN). 
The networks were designed and tested for the classification of 7 different emotions using the solely the FER and KDEF datasets.

There are 4 models that are available for usage:
  1. LeNet
  2. VGG11
  3. VGG13
  4. ResNet

These can be selected as described bellow in the **Instructions**. 

Although the **__init__.py** is a custom file organisation code for the KDEF and FER dataset, it is still possible to train the available models by having the correct directory layout if you are to use a personal dataset:
"sorted_folder"
        "afraid"
        "anger"
        "happy".....

All functionality that are available can be found in the **Instructions** along with the usage.

happy hacking!
## Instructions

### Premilinary Steps
  1. Download the [FER Dataset](https://www.kdef.se/download-2/register.html).
  2.  Download the [FER Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) make sure to retrieve the train.csv file
  **warning** Preferably use the FER dataset for a better performance from the models when training.
  3.  Place the csv file into the folder where main.py is in
  4.  Run __init__.py

### Training a new dataset
  1.  Run the main.py program
  2.  Select the first option
  3.  Go through the steps as followed
  4.  Wait until its finished
  5.  Training data and model files is stored in the **saved** folder

### Resume training on a network
  1.  Run the main.py program
  2.  Select the architecture that you would like to resume training 
  **(NOTE: if the model .pt file you wish to resume training does not match with this option, the program will terminate)**
  3.  Follow through the steps

### Testing exisiting network with the test data
  1. Run the main.py program
  2.  Select the architecture that you would like to resume training 
  **(NOTE: if the model .pt file you wish to resume training does not match with this option, the program will terminate)**
  3.  Follow through the steps
