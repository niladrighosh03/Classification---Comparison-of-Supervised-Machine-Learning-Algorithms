import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load saved models
knn_model = joblib.load('knn_model.pkl')
svm_model = joblib.load('svm_model.pkl')
logreg_model = joblib.load('logreg_model.pkl')
gb_model = joblib.load('gb_model.pkl')
dt_model = joblib.load('dt_model.pkl')
rf_model = joblib.load('rf_model.pkl')
nb_model = joblib.load('nb_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

# Define preprocessing pipeline
num_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_columns = ['origin', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_columns),
        ('cat', cat_transformer, cat_columns)
    ]
)

# Function to preprocess input
def preprocess_input(input_df):
    return preprocessor.fit_transform(input_df)

# Function to make predictions
def predict(input_data):
    processed_data = preprocess_input(input_data)
    predictions = {
        'KNN Model': knn_model.predict(processed_data)[0],
        'SVM Model': svm_model.predict(processed_data)[0],
        'Logistic Regression Model': logreg_model.predict(processed_data)[0],
        'Gradient Boosting Model': gb_model.predict(processed_data)[0],
        'Decision Tree Model': dt_model.predict(processed_data)[0],
        'Random Forest Model': rf_model.predict(processed_data)[0],
        'Naive Bayes Model': nb_model.predict(processed_data)[0],
        'XGBoost Model': xgb_model.predict(processed_data)[0]
    }
    return predictions

# Streamlit UI
st.title("Heart Disease Prediction")

# Input fields
st.header("Enter Patient Data")
id_value = st.text_input("Patient ID")
age = st.number_input("Age", min_value=0, step=1)
origin = st.selectbox("Place of Dataset Study", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])  # Replace with actual locations
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, step=1)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["TRUE", "FALSE"])
restecg = st.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, step=1)
exang = st.selectbox("Exercise-Induced Angina", ["TRUE", "FALSE"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia Type", ["normal", "fixed defect", "reversible defect"])

# Convert input to DataFrame
input_df = pd.DataFrame([[age, origin, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                        columns=num_columns + cat_columns)

if st.button("Predict"):
    predictions = predict(input_df)
    st.write("### Predictions:")
    for model, pred in predictions.items():
        st.write(f"{model}: {'Positive' if pred == 1 else 'Negative'}")


# # Example input fields (modify according to dataset features)
# # Input fields
# st.header("Enter Patient Data")
# id_value = st.text_input("Patient ID")
# age = st.number_input("Age", min_value=0, step=1)
# origin = st.selectbox("Place of Dataset Study", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])  # Replace with actual locations
# sex = st.selectbox("Sex", ["Male", "Female"])
# cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
# trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, step=1)
# chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, step=1)
# fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["TRUE", "FALSE"])
# restecg = st.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"])
# thalch = st.number_input("Maximum Heart Rate Achieved", min_value=0, step=1)
# exang = st.selectbox("Exercise-Induced Angina", ["TRUE", "FALSE"])
# oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, step=0.1)
# slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
# ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
# thal = st.selectbox("Thalassemia Type", ["normal", "fixed defect", "reversible defect"])
