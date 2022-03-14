# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:52:57 2022

@author: Thobela Sixpence
"""


import streamlit as st
import tensorflow as tf

#st.set_option('depreciate.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

# fucntion to load the model
def load_model():
    
    model = tf.keras.models.load_model('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Deep Learning Project 2 - Pneumonia Detection( using Xray Images)/pneumonia_model.hdf5')
    
    return model

model = load_model()
# write heading for the web app
st.title("Pneumonia Detection Web App")
st.write("### Classification Model")


file = st.file_uploader("Please upload a scanner image", type=["jpeg", "png"])


import cv2
from PIL import Image, ImageOps
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input

# create a function to predict
def model_prediction(img, model):
    
    size = (224,224)
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    X = image.img_to_array(img)
    img = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    reshaped_img = img[np.newaxis,...]
    
    classes  = model.predict(reshaped_img)
    
    new_pred = np.argmax(classes, axis=1)
    if new_pred==[1]:
        outcome = "Prediction: Negative Pneumonia"
    else:
        outcome = "Prediction: Positive Pneumonia"
        
    return outcome

if file is None:
    st.text("Please upload a scanner image")
else:
    input_img = Image.open(file)
    st.image(input_img, use_column_width = True)
    prediction = model_prediction(input_img, model)
    st.success(prediction)




