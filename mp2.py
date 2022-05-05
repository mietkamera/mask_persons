#!/usr/local/bin/env/bin/python3

import getopt, sys, os
import pathlib

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
work_dir = str(pathlib.Path(__file__).parent.absolute())
modelFilename = work_dir + '/model/mk_default_model.h5'

# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]
   
# Options
options = "hi:o:m:"
    
# Long options
long_options = ["help", "input=", "output=", "model="]
     
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
                                                                                                                                
        elif currentArgument in ("-m", "--model"):
            if currentValue.startswith('/'):
                modelFilename = currentValue
            else:
                modelFilename = str(cwd) + '/' + currentValue
                                                                                                                                    
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

if not fileExists(modelFilename):
    print ("Model ", modelFilename, " doesn't exist!")
    exit(5)

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

basename = os.path.basename(outputFilename)
dirname = os.path.dirname(outputFilename)
filename, extension = os.path.splitext(basename)

maskFilename = dirname + '/' + filename + '_mask.png'


from tensorflow import keras
from patchify import patchify, unpatchify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random



# import Model
model = keras.models.load_model(modelFilename, compile=False)

large_image = cv2.imread(inputFilename)

#print("Large Image Shape:", large_image.shape)

patches_img = patchify(large_image, (256, 256,3), step=256)

images = np.array(patches_img)

#print("images.shape", images.shape)

images = np.squeeze(images)

#print("images.shape squeeze", images.shape)



pred_patches = []

for i in range(images.shape[0]):
    for j in range(images.shape[1]):
        #print(i,j)
                                
        single_patch = images[i,j,:,:,:]

        #print(single_patch.shape)
        single_patch = (single_patch.astype('float32')) / 255.

        single_patch = np.expand_dims(single_patch, axis=0)
        #print("predict: ", j)
        prediction = model.predict(single_patch)

        pred_patches.append(prediction)


pred_patches = np.array(pred_patches)
#print("pred_patches.shape:",pred_patches.shape)
pred_patches = np.squeeze(pred_patches, axis=1)
#print("pred_patches.shape, squeeze:",pred_patches.shape)

pred_patches = np.reshape(pred_patches,(4, 4, 256, 256, 1))

pred_patches = np.expand_dims(pred_patches, axis=2)
#print("pred_patches.shape", pred_patches.shape)
prediction = unpatchify(pred_patches, (1024, 1024, 1))
#print("prediction.shape - unpatchify", prediction.shape)

prediction_squeeze = np.squeeze(prediction, axis=2)
#print("prediction_squeeze.shape", prediction_squeeze.shape)


prediction_int = prediction_squeeze * 255
prediction_int = prediction_int.astype(np.uint8)


# blur image

blur = cv2.blur(large_image,(15,15),0)
out = large_image.copy()
out[prediction_int>0] = blur[prediction_int>0]

# save images

cv2.imwrite(maskFilename, prediction_int)
cv2.imwrite(outputFilename, out)





