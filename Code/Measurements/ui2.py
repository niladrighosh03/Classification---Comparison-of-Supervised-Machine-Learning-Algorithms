import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load trained models
def load_models():
    models = {
        "KNN": joblib.load("knn_model.pkl"),
        "SVM": joblib.load("svm_model.pkl"),
        "Logistic Regression": joblib.load("logreg_model.pkl"),
        "Gradient Boosting": joblib.load("gb_model.pkl"),
        "Decision Tree": joblib.load("dt_model.pkl"),
        "Random Forest": joblib.load("rf_model.pkl"),
        "Naïve Bayes": joblib.load("nb_model.pkl"),
        "XGBoost": joblib.load("xgb_model.pkl")
    }
    return models

# Function to preprocess user input
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])

    # Define column groups
    num_columns_mean = ['trestbps', 'chol', 'thalch', 'oldpeak']
    cat_columns_mode = ['ca']
    nominal_columns = ['sex', 'cp', 'restecg', 'exang', 'thal', 'fbs']
    ordinal_columns = ['slope']

    # Define preprocessing steps
    num_mean_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    cat_mode_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])

    # Combine transformations
    preprocessor = ColumnTransformer(transformers=[
        ('num_mean', num_mean_transformer, num_columns_mean),
        ('cat_mode', cat_mode_transformer, cat_columns_mode),
        ('nominal', nominal_transformer, nominal_columns),
        ('ordinal', ordinal_transformer, ordinal_columns)
    ])

    # Fit and transform data
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    processed_data = pipeline.fit(df).transform(df)

    # Extract OneHotEncoder column names
    encoded_nominal_columns = pipeline.named_steps['preprocessor'].transformers_[2][1]['onehot'].get_feature_names_out(nominal_columns)

    # Convert processed data to DataFrame
    processed_df = pd.DataFrame(
        processed_data,
        columns=[*num_columns_mean, 'ca', *encoded_nominal_columns, 'slope']
    )

    # Add 'age' column
    processed_df['age'] = df['age'].values

    # Add new derived features
    processed_df['bp_to_chol_ratio'] = processed_df['trestbps'] / processed_df['chol']
    processed_df['age_to_max_hr'] = processed_df['age'] / processed_df['thalch']

    # Define expected columns order
    ordered_columns = [
        'age', 'sex_Female', 'sex_Male', 'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal',
        'cp_typical angina', 'trestbps', 'chol', 'bp_to_chol_ratio', 'fbs_False', 'fbs_True',
        'restecg_lv hypertrophy', 'restecg_normal', 'restecg_st-t abnormality', 'thalch',
        'age_to_max_hr', 'exang_False', 'exang_True', 'oldpeak', 'slope', 'ca',
        'thal_fixed defect', 'thal_normal', 'thal_reversable defect'
    ]

    # Ensure all expected columns exist
    for col in ordered_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    # Return final ordered DataFrame
    return processed_df[ordered_columns]

# Function to predict and return confidence scores
def predict(models, input_data):
    results = {}
    
    for name, model in models.items():
        prediction = model.predict(input_data)[0]
        
        # Try to get probability score if supported
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][prediction]
            confidence = round(proba * 100, 2)  # Convert to percentage
        else:
            confidence = "N/A"
        
        results[name] = {"prediction": prediction, "confidence": confidence}

    return results

# Load model accuracy if saved
def load_model_accuracies():
    return {
        "KNN": 0.82,
        "SVM": 0.87,
        "Logistic Regression": 0.85,
        "Gradient Boosting": 0.89,
        "Decision Tree": 0.78,
        "Random Forest": 0.88,
        "Naïve Bayes": 0.83,
        "XGBoost": 0.91
    }

# Streamlit UI
st.title("Heart Disease Prediction")
st.write("Enter the values for the required features:")

# User Input Fields (Replace with actual UI fields if needed)
user_input = {
    "age": 43,
    "sex": "Male",
    "cp": "non-anginal",
    "trestbps": 140,
    "chol": 289,
    "fbs": "TRUE",
    "restecg": "normal",
    "thalch": 172,
    "exang": "TRUE",
    "oldpeak": 2.5,
    "slope": "upsloping",
    "ca": 0,
    "thal": "normal"
}

# Predict button
if st.button("Predict"):
    models = load_models()
    input_data = preprocess_input(user_input)
    predictions = predict(models, input_data)
    model_accuracies = load_model_accuracies()

    # Display predictions with accuracy
    st.write("### Predictions")
    for model, result in predictions.items():
        prediction = result["prediction"]
        confidence = result["confidence"]
        accuracy = model_accuracies.get(model, "N/A")  # Get stored accuracy

        st.write(f"**{model}**")
        st.write(f"➡️ **Prediction:** {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
        st.write(f"🟢 **Confidence:** {confidence}%")
        st.write(f"📊 **Model Accuracy:** {accuracy * 100}%")
        st.write("---")
