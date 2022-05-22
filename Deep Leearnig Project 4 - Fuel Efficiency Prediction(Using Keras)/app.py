# import libraries
from flask import Flask, render_template, url_for, request, jsonify
import joblib
import os
import pickle
import numpy as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

app = Flask(__name__)

# the route for the web app
@app.route("/")
# this is the home page
def index():
    return render_template('home.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    # get the input from the user
    cylinders = int(request.form["cylinders"])
    displacement = int(request.form["displacement"])
    horsepower = int(request.form["horsepower"])
    weight = int(request.form["weight"])
    acceleration = int(request.form["acceleration"])
    model_year = int(request.form["model_year"])
    origin = int(request.form["origin"])

    values = [[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]]

    scaler_path = os.path.join(os.path.dirname('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Deep Leearnig Project 4 - Fuel Efficiency Prediction/models/'),
                               'scaler.pkl')

    sc = None
    with open(scaler_path, 'rb') as f:
        sc = pickle.load(f)

    values = sc.transform(values)

    model = load_model(r"C:\Users\Cash Crusaders\Desktop\My Portfolio\Projects\Data Science Projects\Deep Leearnig Project 4 - Fuel Efficiency Prediction\models\fuel_model.h5")

    prediction = model.predict(values)
    prediction = float(prediction)
    print(prediction)

# km per hr
    prediction = prediction/2.352
    json_dict = {
        "prediction km per ltr": prediction
    }

    return jsonify(json_dict)


if __name__ == "__main__":
    app.run(debug=True, port=3298)
