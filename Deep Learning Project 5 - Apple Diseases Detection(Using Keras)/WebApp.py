# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 04:03:47 2022

@author: Thobela Sixpence
"""

import streamlit as st
import tensorflow as tf

#st.set_option('depreciate.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

# fucntion to load the model
def load_model():
    
    model = tf.keras.models.load_model('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Deep Learning Project 5 - Apple Diseases Detection(Using Keras)/models/apple_model.h5')
    
    return model

model = load_model()
# write heading for the web app
st.title("Folliar Disease Detection Web App")
st.write("### Classification Model")


file = st.file_uploader("Please upload a scanner image", type=["jpeg", "png"])

import cv2
from PIL import Image, ImageOps
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

# create a function to predict
def model_prediction(img_path, model):
    '''takes in path of an image and model in order to predict the resulting group this image belongs to'''
    test_img = image.load_img(img_path, target_size=(224, 224))
    test_img = image.img_to_array(test_img)
    test_img = test_img/225
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    
    categories = ['healthy','multiple_diseases','rust','scab']
    
    class_num = np.argmax(result)
    if class_num==0:
        outcome = "You have been diagnosed of " + categories[class_num]
    elif class_num==1:
        outcome = "You have been diagnosed of " + categories[class_num]
    elif class_num==2:
        outcome = "You have been diagnosed of " + categories[class_num]
    elif class_num==3:
         outcome = "You have been diagnosed of " + categories[class_num]
    else:
        outcome = "Not found."
        
    return result

if file is None:
    st.text("Please upload a scanner image")
else:
    input_img = Image.open(file)
    st.image(input_img, use_column_width = True)
    prediction = model_prediction(input_img, model)
    st.success(prediction)


