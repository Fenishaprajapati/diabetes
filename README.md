# Diabetes Prediction Web Application

## Project Overview
This project is a machine learning-based web application developed using Flask that predicts the likelihood of diabetes in patients based on their medical information. The main goal is to provide an accessible tool for early diabetes diagnosis, which can be useful in healthcare scenarios to aid in quick screening and potentially early detection.

## Dataset
The dataset used for this project is the PIMA Indian Diabetes Dataset. It contains the following features:

- Pregnancies: Number of times the patient has been pregnant
- Glucose: Plasma glucose concentration (mg/dL)
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (µU/mL)
- BMI: Body Mass Index (weight in kg/height in m²)
- DiabetesPedigreeFunction: A function that scores the likelihood of diabetes based on family history
- Age: Age of the patient (years)
- Outcome: 0 or 1, where 1 indicates the presence of diabetes and 0 indicates absence


### 1. BMI (Body Mass Index)
Minimum: ~15 (often seen in cases of extreme underweight or malnutrition)
Maximum: 60+ (rare but possible in extreme obesity cases)

### 2. Glucose Level (mg/dL)
Minimum: ~40 (low blood sugar, or hypoglycemia, which can be dangerous)
Maximum: ~600+ (high blood sugar, or hyperglycemia, especially in unmanaged diabetes cases)

### 3. Blood Pressure (Systolic/Diastolic in mm Hg)
Minimum: Systolic ~70 / Diastolic ~40 (seen in cases of severe hypotension)
Maximum: Systolic ~250 / Diastolic ~140+ (seen in cases of hypertensive crisis)

### 4. Age (Years)
Minimum: Newborns (0 years)
Maximum: 120+ years (in rare cases)

### 5. Insulin Level (µU/mL)
Minimum: 0 (in cases where the body produces little or no insulin, as in type 1 diabetes)
Maximum: 300+ µU/mL (in severe cases of insulin resistance or high-dose insulin therapy)

### 6. Skin Thickness (mm)
Minimum: ~1 mm (thin skin measurement in lean individuals)
Maximum: 99+ mm (rarely above this in obesity cases)

### 7. Diabetes Pedigree Function
This is a calculated value based on family history of diabetes, so there isn’t a real-life “limit.” Values usually range from 0 to 2.5, but may exceed this in some cases.

### - 8. Pregnancies
Minimum: 0 (no pregnancies)
Maximum: 20+ (though very rare)

- Dataset link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## Tech Stack
- Programming Language: Python
- Web Framework: Flask
- Machine Learning Libraries: scikit-learn, pandas, numpy
- Data Handling: pandas, numpy
- Deployment: Flask server

## Model Building and Evaluation
The following machine learning models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- The evaluation metrics used were:

- Accuracy: The percentage of correctly predicted outcomes.
- Precision: The ratio of true positive predictions to the total positive predictions.
- Recall: The ratio of true positive predictions to the total actual positives.
- F1 Score: The harmonic mean of precision and recall.
- The best performing model was selected based on the F1 Score to ensure a good - balance between precision and recall.


## Error Handling and Logging
- CustomException: A custom exception class is used to handle and log errors in a detailed manner.
- Logging: All actions, such as data ingestion, model training, and prediction, are logged to a file for monitoring and debugging purposes.
