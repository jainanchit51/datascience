import numpy as np
from numpy import array
import pandas as pd
from oneoone import LogisticRegression
import pickle
from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)


def vectrorize(text_arr,dictt):
    vectors  = []
    for sentence in text_arr:
        vector = [0] * len(dictt)
        words = (sentence.lower().strip().split(" "))
        for word in words:
            index = dictt[word]
            vector[index] += 1
        vectors.append(vector)
    return vectors


@app.route("/predict")
def predict():
    text_arr = []
    text_arr.append(request.args.get('text'))
    file = open("dictionary.pickle",'rb')
    dictt = pickle.load(file)
    prediction_vector = np.array(vectrorize(text_arr, dictt))

    lr=0.001
    num_iter=10000
    logr = LogisticRegression(lr,num_iter, 0.5)

    prediction = logr.predict(np.array(prediction_vector))[0]
    return str(prediction)
