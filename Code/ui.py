import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load saved models
model1 = joblib.load('model1.pkl')  # Replace with actual filenames
model2 = joblib.load('model2.pkl')
model3 = joblib.load('model3.pkl')

# Define preprocessing pipeline
num_columns_mean = ['trestbps', 'chol', 'thalach', 'oldpeak']
cat_columns_mode = ['ca']
nominal_columns = ['sex', 'cp', 'restecg', 'exang', 'thal', 'fbs']
ordinal_columns = ['slope']

num_mean_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
cat_mode_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])
nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num_mean', num_mean_transformer, num_columns_mean),
        ('cat_mode', cat_mode_transformer, cat_columns_mode),
        ('nominal', nominal_transformer, nominal_columns),
        ('ordinal', ordinal_transformer, ordinal_columns)
    ]
)

# Function to preprocess input
def preprocess_input(input_df):
    return preprocessor.fit_transform(input_df)

# Function to make predictions
def predict(input_data):
    processed_data = preprocess_input(input_data)
    predictions = {
        'Model 1': model1.predict(processed_data)[0],
        'Model 2': model2.predict(processed_data)[0],
        'Model 3': model3.predict(processed_data)[0]
    }
    return predictions

# Streamlit UI
st.title("Heart Disease Prediction")

# Input fields
st.header("Enter Patient Data")
columns = num_columns_mean + cat_columns_mode + nominal_columns + ordinal_columns
input_data = []
for col in columns:
    value = st.number_input(f"{col}", min_value=0.0, step=0.1)
    input_data.append(value)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data], columns=columns)

if st.button("Predict"):
    predictions = predict(input_df)
    st.write("### Predictions:")
    for model, pred in predictions.items():
        st.write(f"{model}: {'Positive' if pred == 1 else 'Negative'}")
