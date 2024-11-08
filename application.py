import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

#Route for a home page

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')
    else:
        data=CustomData(
            pregnancies=int(request.form.get('Pregnancies')),
            glucose=int(request.form.get('Glucose')),
            bloodPressure=int(request.form.get('BloodPressure')),
            skinThickness=int(request.form.get('SkinThickness')),
            insulin=int(request.form.get('Insulin')),
            bmi=float(request.form.get('BMI')),
            diabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction')),
            age=int(request.form.get('Age'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print("after Prediction")

        rounded_result = round(results[0], 0)
        
        return render_template('home.html', results=rounded_result)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        

# while we are deploying we always remove debug=True    