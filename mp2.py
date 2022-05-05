#!/usr/local/bin/env/bin/python3

import getopt, sys, os

def fileExists(filename):

    try:
        with open(filename, 'r') as fh:
            return True

    except FileNotFoundError:
        return False
    except IsADirectoryError:
        return True


cwd = os.getcwd() 
inputFilename = ''
outputFilename = ''

# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]
   
# Options
options = "hi:o:"
    
# Long options
long_options = ["help", "input=", "output="]
     
try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)
                     
    # checking each argument
    for currentArgument, currentValue in arguments:
                                   
        if currentArgument in ("-h", "--help"):
            print ("Displaying Help")
                                                                       
        elif currentArgument in ("-i", "--input"):
            if currentValue.startswith('/'):
                inputFilename = currentValue
            else:
                inputFilename = str(cwd) + '/' + currentValue
            if not fileExists(inputFilename):
                print (inputFilename, " doesn't exist!")
                exit(1)
                                                                                                            
        elif currentArgument in ("-o", "--output"):
            if currentValue.startswith('/'):
                outputFilename = currentValue
            else:
                outputFilename = str(cwd) + '/' + currentValue
            if fileExists(outputFilename):
                print (outputFilename, " already exists!")
                exit(2)
                                                                                                                                
                                                                                                                                    
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    exit()


if len(argumentList) == 0:
    print("Missing argument")
    exit(3)

if inputFilename == '':
    print("Input-Filename not specified")
    exit(4)

if outputFilename == '':
    print("Output-Filename not specified")
    exit(4)




from tensorflow import keras
from patchify import patchify, unpatchify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import pathlib



# Get the current working 
# directory (CWD) 
    





work_dir = str(pathlib.Path(__file__).parent.absolute())

model_file = work_dir + '/model/mk_default_model.h5'

#model_file = '/content/drive/MyDrive/ML/mk_model_transfer_learning_11.h5'
#imgs_path = '/content/drive/MyDrive/ML/imgs_test_2/resize/'
imgs_path = work_dir + '/'
#output_path = '/content/drive/MyDrive/ML/prediction_tmp/'
#output_path = '/content/drive/MyDrive/ML/imgs_test_2/prediction/'
output_path = work_dir + '/prediction/'

#filename = work_dir + '/test.png'



# import Model
model = keras.models.load_model(model_file, compile=False)

#image_list = sorted(glob.glob(imgs_path + "*.png"))

#if len(image_list) > 0:
#      image_number = random.randint(0, len(image_list)-1)
##else:
#        image_number = 0

#image_path = image_list[image_number]
        #image_path = image_list[0]

image_path = filename

large_image = cv2.imread(image_path)

#print("Large Image Shape:", large_image.shape)

patches_img = patchify(large_image, (256, 256,3), step=256)

images = np.array(patches_img)

#print("images.shape", images.shape)

images = np.squeeze(images)

#print("images.shape squeeze", images.shape)

