# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:57:15 2022

@author: Suji

"""

import pickle
from flask import Flask, request
import pandas as pd
import numpy as np
import flasgger
from flasgger import Swagger

with open(r"C:\Users\Pavi\OneDrive\Desktop\Logistic_Model\model_log_pkl",'rb') as model_svm_pickle:
    model_svm = pickle.load(model_svm_pickle)
    

    
ml_api = Flask(__name__)
swagger = Swagger(ml_api)


@ml_api.route('/predict', methods=['Get'])
def predict_svc():    
    """Let's Predict
    Heart Disease Prediction Using Logistic Regression Model
    ---
    parameters:  
      - name: age
        in: query
        type: number
        required: true
      - name: currentSmoker(yes-1,no-0)
        in: query
        type: number
        required: true
      - name: cigsPerDay
        in: query
        type: number
        required: true
      - name: BPMeds
        in: query
        type: number
        required: true
      - name:  prevalentStroke(yes-1,no-0)
        in: query
        type: number
        required: true
      - name: prevalentHyp(yes-1,no-0)
        in: query
        type: number
        required: true
      - name: diabetes(yes-1,no-0)
        in: query
        type: number
        required: true
      - name: totChol
        in: query
        type: number
        required: true
      - name: BMI
        in: query
        type: number
        required: true
      - name: heartRate
        in: query
        type: number
        required: true
      - name: glucose
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    age= request.args.get('age')
    currentSmoker= request.args.get('currentSmoker')
    cigsPerDay = request.args.get('cigsPerDay')
    BPMeds= request.args.get('BPMeds')
    prevalentStroke = request.args.get('prevalentStroke')
    prevalentHyp = request.args.get('prevalentHyp')
    diabetes = request.args.get('diabetes')
    totChol = request.args.get('totChol')
    BMI = request.args.get('BMI')
    heartRate = request.args.get('heartRate')
    glucose= request.args.get('glucose')
    
    
    input_data = np.array([[age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,BMI,heartRate,glucose]])
    prediction = model_svm.predict(input_data)
    return str(prediction)



if __name__ == '__main__':
    ml_api.run()

