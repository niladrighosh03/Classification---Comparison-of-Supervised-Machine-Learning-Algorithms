import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import time  # For progress bar effect

# Custom Styling
st.markdown(
    """
    <style>
        .stButton>button { background-color: #007bff; color: white; font-size: 18px; border-radius: 8px; padding: 10px 20px; }
        .stButton>button:hover { background-color: #0056b3; }
        .result-box { padding: 10px; border-radius: 8px; text-align: center; font-size: 18px; }
        .positive { background-color: #ffcccc; color: #b30000; }
        .negative { background-color: #ccffcc; color: #006600; }
    </style>
    """,
    unsafe_allow_html=True
)

# Load trained models and their accuracies
def load_models():
    models = {
        "KNN": (joblib.load("knn_model.pkl"), 0.85),
        "SVM": (joblib.load("svm_model.pkl"), 0.88),
        "Logistic Regression": (joblib.load("logreg_model.pkl"), 0.86),
        "Gradient Boosting": (joblib.load("gb_model.pkl"), 0.90),
        "Decision Tree": (joblib.load("dt_model.pkl"), 0.82),
        "Random Forest": (joblib.load("rf_model.pkl"), 0.89),
        "Naïve Bayes": (joblib.load("nb_model.pkl"), 0.81),
        "XGBoost": (joblib.load("xgb_model.pkl"), 0.91)
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

    return processed_df[ordered_columns]

# Function to predict using all models
def predict(models, input_data):
    predictions = {name: (model.predict(input_data)[0], acc) for name, (model, acc) in models.items()}
    return predictions

# Streamlit UI
st.title("🔬 Heart Disease Prediction System")
st.markdown("### 🏥 A Machine Learning-Based Health Assistant")

# Sidebar for User Input
st.sidebar.header("User Input Parameters")
age = st.sidebar.number_input("Age", 18, 100, 45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.sidebar.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"])
thalch = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise-Induced Angina", ["True", "False"])
oldpeak = st.sidebar.number_input("ST Depression Induced", 0.0, 6.2, 1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
ca = st.sidebar.number_input("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
thal = st.sidebar.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

user_input = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang, 
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}



# Define class labels
class_labels = {
    0: "No Heart Disease (0)",
    1: "Mild Disease (1)",
    2: "Moderate Disease (2)",
    3: "Severe Disease (3)",
    4: "Critical Condition (4)"
}

if st.button("🔍 Predict"):
    with st.spinner("Processing... Please wait..."):
        time.sleep(2)  # Simulated loading effect
        models = load_models()
        input_data = preprocess_input(user_input)
        predictions = predict(models, input_data)

    # Find the model with highest & lowest accuracy
    best_model = max(predictions, key=lambda x: predictions[x][1])
    worst_model = min(predictions, key=lambda x: predictions[x][1])

    # Display results
    st.markdown("## 🔎 Prediction Results")

    for model, (prediction, accuracy) in predictions.items():
        class_name = class_labels.get(prediction, f"Unknown ({prediction})")

        # Highlight highest & lowest accuracy models with colored boxes
        if model == best_model:
            st.markdown(
                f'<div style="background-color:#218838; color:white; padding:10px; border-radius:5px; border-left: 5px solid #1E7E34;">'
                f'🎉 <b>Best Model: {model}</b><br>'
                f'🏥 Prediction: <b>{class_name}</b><br>'
                f'🎯 Accuracy: <b>{accuracy * 100:.2f}%</b>'
                f'</div>',
                unsafe_allow_html=True
            )
        elif model == worst_model:
            st.markdown(
                f'<div style="background-color:#C82333; color:white; padding:10px; border-radius:5px; border-left: 5px solid #A71D2A;">'
                f'⚠️ <b>Least Accurate Model: {model}</b><br>'
                f'🏥 Prediction: <b>{class_name}</b><br>'
                f'🎯 Accuracy: <b>{accuracy * 100:.2f}%</b>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"### **{model}**")
            st.write(f"- 🏥 Prediction: **{class_name}**")
            st.write(f"- 🎯 Accuracy: **{accuracy * 100:.2f}%**")
        
        st.write("---")  # Separator for readability
