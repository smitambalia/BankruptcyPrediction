
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from flask import Flask , jsonify, request
from flasgger import Swagger
import flask

app = Flask(__name__)
Swagger(app)

@app.route("/")
def welcome():
    return "Welcome to the Bankruptcy Prediction"
    
@app.route("/bankruptOne")
def bankruptOne():
    return flask.render_template("bankruptOne.html")

@app.route("/bankruptMany")
def bankruptMany():
    return flask.render_template("bankruptMany.html")

@app.route("/predictBankruptcy", methods=['POST'])
def predictOne():
    '''
    Predicting the bankrupcy for one data only with this function
    '''
    test_data = request.form.to_dict()
    test_data = pd.DataFrame([test_data])
    scalar = joblib.load('trainscalar.pkl')
    test_data = scalar.transform(test_data)

    test = xgb.DMatrix(test_data)
    clf = xgb.Booster({'nthread':4})
    clf.load_model('bankruptpred.model')

    pred = clf.predict(test)
    best_pred = str(np.asarray(np.argmax(pred)))
    return jsonify({"Bankruptcy:":best_pred})


@app.route("/predictManyBankruptcy",methods=['POST'])
def predictMany():
    '''
    Predicting the bankrupcy for bulk data on this function
    '''
    test_data = pd.read_excel(request.files.get("test"))
    scalar = joblib.load('trainscalar.pkl')
    test_data = scalar.transform(test_data)

    test = xgb.DMatrix(test_data)
    clf = xgb.Booster({'nthread':4})
    clf.load_model('bankruptpred.model')

    preds = clf.predict(test)
    pred_list = np.asarray([np.argmax(pred) for pred in preds])
    # return str(list(pred_list))
    return jsonify({"bankruptcy" : list(str(pred_list))})

if(__name__== '__main__'):
    app.run(host='0.0.0.0',port=8000)
