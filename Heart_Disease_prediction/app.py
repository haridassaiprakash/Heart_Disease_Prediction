import numpy as np
import pickle
import pandas as pd
import sklearn

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict', methods=["POST", 'GET'])
def results():
    Chestpaintype = float(request.form['Chestpaintype'])
    bP = float(request.form['bP'])
    Cholesterol = float(request.form['Cholesterol'])
    FBSover120 = float(request.form['FBSover120'])
    EKGresults = float(request.form['EKGresults'])
    Thallium = float(request.form['Thallium'])
    Numberofvesselsfluro = float(request.form['Numberofvesselsfluro'])
    STdepression = float(request.form['STdepression'])
    SlopeofST = float(request.form['SlopeofST'])
    

    X = np.array([[Chestpaintype, bP, Cholesterol, FBSover120, EKGresults, Thallium, Numberofvesselsfluro, STdepression, SlopeofST]])

    model = pickle.load(open('model.pkl', 'rb'))
    Y_prediction = model.predict(X)

    return jsonify({'Prediction': float(Y_prediction)})


if __name__ == '__main__':
    app.run(debug=True, port=1234)
