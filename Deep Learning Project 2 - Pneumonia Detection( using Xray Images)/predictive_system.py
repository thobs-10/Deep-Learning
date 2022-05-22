# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:46:21 2022

@author: Thobela Sixpence
"""

import tensorflow as tf
from keras.preprocessing import image
model = tf.rkeras.models.load_model('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Deep Learning Project 2 - Pneumonia Detection( using Xray Images)/pneumonia_model.hdf5')
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np


# take a random image and do the prediction
img = image.load_img(r'C:\Users\Cash Crusaders\Desktop\My Portfolio\Projects\Data Science Projects\Deep Learning Project 1  - Corona Virus Detection ( Using Chest Xray Images)\dataset\Normal\IM-0178-0001.jpeg',target_size=(224, 224))
img_plot = plt.imshow(img)

# convert the image into an array
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)

# proprocess the image
img_data = preprocess_input(X)

classes  = model.predict(img_data)
new_pred = np.argmax(classes, axis=1)
if new_pred==[1]:
    print("Prediction: Normal")
else:
    print("Prediction: Pneumonia")

