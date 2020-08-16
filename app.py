# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 19:11:59 2020

@author: admin
"""


import numpy as np
from flask import Flask, request, render_template, url_for
import pickle
import joblib


app = Flask(__name__)
model = joblib.load("Students_mark_predictor_model.pkl")
model
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    value = np.array(input_features)
    output = model.predict([value])[0][0].round(2)
    return render_template('index.html', Prediction_text = f"you will get {output}% marks, when you do study {input_features} hours per day")
if __name__ == "__main__":
    app.run(debug=True)