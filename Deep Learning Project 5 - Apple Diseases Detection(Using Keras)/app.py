# import libraries
from flask import Flask, render_template, url_for, request, jsonify
import joblib
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# define the flask app
app = Flask(__name__)

# load model
model = load_model(r'C:\Users\Cash Crusaders\Desktop\My Portfolio\Projects\Data Science Projects\Deep Learning Project 5 - Apple Diseases Detection(Using Keras)\models\apple_model.h5')

# model prediction
def model_prediction(img_path, model):
    '''takes in path of an image and model in order to predict the resulting group this image belongs to'''
    test_img = image.load_img(img_path, target_size=(224, 224))
    test_img = image.img_to_array(test_img)
    test_img = test_img/225
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    '''upload an image to be predicted'''
    if request.method == 'POST':
        # get the file from post request
        f = request.files['file']

        # save the file to uploads folder
        basepath = os.path.dirname(os.path.realpath('__file__'))
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_prediction(file_path, model)

        categories = ['Healthy', 'Multiple Disease', 'Rust', 'Scab']
        # process your result for human
        pred_class = result.argmax()
        output = categories[pred_class]
        return output
    return None

if __name__=='__main__':
    app.run(debug=False, port=5926)


