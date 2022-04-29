####!/usr/local/bin/env/bin/python3

from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from patchify import patchify, unpatchify



import pathlib

model_path = pathlib.Path(__file__).parent.absolute() + "/model/mk_default_model.h5"

# import Model


model = keras.models.load_model(model_path, compile=False)


