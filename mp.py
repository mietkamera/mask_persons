#!/usr/local/bin/env/bin/python3
# Project: How To Detect Objects in an Image Using Semantic Segmentation
# Author: Addison Sears-Collins
# Date created: February 24, 2021
# Description: A program that classifies pixels in an image. The real-world
#   use case is autonomous vehicles. Uses the ENet neural network architecture.
 
import cv2 # Computer vision library
import numpy as np # Scientific computing library 
import os # Operating system library 
import imutils # Image processing library
 
#ORIG_IMG_FILE = 'test_image_1.jpg'
ORIG_IMG_FILE = 'test_images/c7b056_20210413142002.jpg'
ENET_DIMENSIONS = (1024, 512) # Dimensions that ENet was trained on
RESIZED_WIDTH = 600
IMG_NORM_RATIO = 1 / 255.0 # In grayscale a pixel can range between 0 and 255
 
# Read the image
input_img = cv2.imread(ORIG_IMG_FILE)
 
# Resize the image while maintaining the aspect ratio
print("Resizing image...")
input_img = imutils.resize(input_img, width=RESIZED_WIDTH)
 
# Create a blob. A blob is a group of connected pixels in a binary 
# image that share some common property (e.g. grayscale value)
# Preprocess the image to prepare it for deep learning classification
print("Creating blob...")
input_img_blob = cv2.dnn.blobFromImage(input_img, IMG_NORM_RATIO,
  ENET_DIMENSIONS, 0, swapRB=True, crop=False)
     
# Load the neural network (i.e. deep learning model)
print("Loading the neural network...")
enet_neural_network = cv2.dnn.readNet('./enet-cityscapes/enet-model.net')
 
# Set the input for the neural network
enet_neural_network.setInput(input_img_blob)
 
# Get the predicted probabilities for each of the classes (e.g. car, sidewalk)
# These are the values in the output layer of the neural network
enet_neural_network_output = enet_neural_network.forward()
 
# Load the names of the classes
class_names = (
  open('./enet-cityscapes/enet-classes.txt').read().strip().split("\n"))
 
# Print out the shape of the output
# (1, number of classes, height, width)
#print(enet_neural_network_output.shape)
 
# Extract the key information about the ENet output
(number_of_classes, height, width) = enet_neural_network_output.shape[1:4] 
 
# number of classes x height x width
print(enet_neural_network_output.shape)
 
# Find the class label that has the greatest probability for each image pixel
class_map = np.argmax(enet_neural_network_output[0], axis=0)
class_prob = np.max(enet_neural_network_output[0], axis=0)
print(class_map[122][1021])
print(class_prob[122][1021])
print(enet_neural_network_output[0][11][122][1021])

# Load a list of colors. Each class will have a particular color. 
if os.path.isfile('./enet-cityscapes/enet-colors.txt'):
  IMG_COLOR_LIST = (
    open('./enet-cityscapes/enet-colors.txt').read().strip().split("\n"))
  IMG_COLOR_LIST = [np.array(color.split(",")).astype(
    "int") for color in IMG_COLOR_LIST]
  IMG_COLOR_LIST = np.array(IMG_COLOR_LIST, dtype="uint8")
     
# If the list of colors file does not exist, we generate a 
# random list of colors
else:
  np.random.seed(1)
  IMG_COLOR_LIST = np.random.randint(0, 255, size=(len(class_names) - 1, 3),
    dtype="uint8")
  IMG_COLOR_LIST = np.vstack([[0, 0, 0], IMG_COLOR_LIST]).astype("uint8")
   
person = (class_map == 12) 
print(class_map.shape)
print(person[122])

# Tie each class ID to its color
# This mask contains the color for each pixel. 
#class_map_mask = IMG_COLOR_LIST[class_map]
class_map_mask = IMG_COLOR_LIST[class_map * person * (class_prob>7)]
 
print(class_map_mask[122][1021])

# We now need to resize the class map mask so its dimensions
# is equivalent to the dimensions of the original image
class_map_mask = cv2.resize(class_map_mask, (
  input_img.shape[1], input_img.shape[0]),
    interpolation=cv2.INTER_NEAREST)

 
# Overlay the class map mask on top of the original image. We want the mask to
# be transparent. We can do this by computing a weighted average of
# the original image and the class map mask.
enet_neural_network_output = ((0.61 * class_map_mask) + (
  0.39 * input_img)).astype("uint8")

#enet_neural_network_output = (class_map_mask +
#  input_img).astype("uint8")

# Create a legend that shows the class and its corresponding color
class_legend = np.zeros(((len(class_names) * 25) + 25, 300, 3), dtype="uint8")
     
# Put the class labels and colors on the legend
for (i, (cl_name, cl_color)) in enumerate(zip(class_names, IMG_COLOR_LIST)):
  color_information = [int(color) for color in cl_color]
  cv2.putText(class_legend, cl_name, (5, (i * 25) + 17),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  cv2.rectangle(class_legend, (100, (i * 25)), (300, (i * 25) + 25),
                  tuple(color_information), -1)
 
# Combine the original image and the semantic segmentation image
combined_images = np.concatenate((input_img, enet_neural_network_output), axis=1) 
 
# Resize image if desired
#combined_images = imutils.resize(combined_images, width=1000)
 
# Display image
#cv2.imshow('Results', enet_neural_network_output) 
cv2.imshow('Results', combined_images) 
cv2.imshow("Class Legend", class_legend)
print(combined_images.shape)
cv2.waitKey(0) # Display window until keypress
cv2.destroyAllWindows() # Close OpenCV
