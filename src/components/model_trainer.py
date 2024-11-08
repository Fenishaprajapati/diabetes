import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Support Vector Machine": SVC()
            }

            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                },
                "Support Vector Machine": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "gamma": ["scale", "auto"]
                }
            }

            # Evaluate models
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                           models=models, param=params)

            # Find the best model based on F1 Score
            best_model_name = max(model_report, key=lambda k: model_report[k]["F1 Score"])
            best_model_score = model_report[best_model_name]["F1 Score"]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No satisfactory model found (F1 Score < 0.6)")

            logging.info(f"Best model found: {best_model_name} with F1 Score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate on test data and return F1 Score as performance metric
            y_test_pred = best_model.predict(X_test)
            f1 = f1_score(y_test, y_test_pred)
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)

            # Print model performance
            # print(f"Best Model: {best_model_name}")
            # print(f"F1 Score: {f1}")
            # print(f"Accuracy: {accuracy}")
            # print(f"Precision: {precision}")
            # print(f"Recall: {recall}")

            return {
                "Model": best_model_name,
                "F1 Score": f1,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall
            }

        except Exception as e:
            raise CustomException(e, sys)
