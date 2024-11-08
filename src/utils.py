import os
import sys
import dill
import pickle

import numpy as np 
import pandas as pd

from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            para = param[model_name]
            
            # Perform GridSearchCV
            gs = GridSearchCV(model, para, cv=3, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)

            # Set model with the best parameters and train
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)

            # Store metrics for each model
            report[model_name] = {
                "Best Params": gs.best_params_,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)