import pickle
from flask import Flask,request,url_for,render_template,jsonify
import numpy as np
import pandas as pd


app=Flask(__name__)

## load the model
regmodel = pickle.load(open('regression.pkl','rb'))
scalar = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_data_scaled = scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = regmodel.predict(new_data_scaled)[0]
        return render_template('home.html',results="The Fire Weather Index Prediction is : {}".format(result))
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
