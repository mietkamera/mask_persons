<<<<<<< HEAD
#!/usr/local/bin/env/bin/python3
=======
####!/usr/local/bin/env/bin/python3
>>>>>>> cd06120e9747bf0d886a3a8d0baa7eee3c45bf0b

from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from patchify import patchify, unpatchify



import pathlib

<<<<<<< HEAD
model_file = str(pathlib.Path(__file__).parent.absolute()) + '/model/mk_default_model.h5'
=======
model_path = pathlib.Path(__file__).parent.absolute() + "/model/mk_default_model.h5"
>>>>>>> cd06120e9747bf0d886a3a8d0baa7eee3c45bf0b

# import Model


<<<<<<< HEAD
model = keras.models.load_model(model_file, compile=False)
=======
model = keras.models.load_model(model_path, compile=False)
>>>>>>> cd06120e9747bf0d886a3a8d0baa7eee3c45bf0b


