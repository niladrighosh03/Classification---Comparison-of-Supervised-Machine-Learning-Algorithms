
# ❤️ Heart Disease Prediction Using Machine Learning

This project applies supervised machine learning algorithms to predict the risk of heart disease using the UCI Heart Disease dataset. The dataset consists of patient attributes and clinical measurements, which are used to classify individuals into heart disease risk categories. 

The implementation includes:  
- 🛠️ Data preprocessing  
- 📊 Exploratory analysis  
- 🤖 Model training and evaluation  
- 🌐 A Streamlit-based deployment for real-time predictions.

---

## 📜 Table of Contents

1. [📋 Overview](#overview)  
2. [📈 Dataset](#dataset)  
3. [🛠️ Preprocessing](#preprocessing)  
4. [🤖 Algorithms](#algorithms)  
5. [🚀 Usage](#usage)  
6. [🔮 Future Work](#future-work)  
7. [🙌 Acknowledgments](#acknowledgments)  

---

## 📋 Overview

Heart disease remains a leading cause of mortality worldwide. Accurate prediction models can assist in early detection and prevention. This project evaluates multiple machine learning algorithms and identifies the best-performing model to classify heart disease risk. Results show that ensemble methods like XGBoost and Gradient Boosting outperform others in accuracy and reliability.

---

## 📈 Dataset

### Source  
[UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

### Features  
- **Demographics**: Age, Sex  
- **Clinical Measurements**: Resting blood pressure, serum cholesterol, maximum heart rate, etc.  
- **Categorical Attributes**: Chest pain type, thalassemia, slope of the ST segment  

### Target Variable  
The presence of heart disease, categorized into five classes (0–4).

---

## 🛠️ Preprocessing

The dataset underwent extensive preprocessing:  
- ✨ **Handling Missing Values**: Imputation with mode (categorical) and mean (numerical) values.  
- 📉 **Outlier Detection**: Addressed using interquartile range (IQR).  
- 🧩 **Feature Engineering**: Added derived features like `BP-to-Cholesterol Ratio`.  
- 📏 **Scaling and Encoding**: Applied `StandardScaler` and one-hot encoding for categorical variables.

---

## 🤖 Algorithms

The following machine learning models were evaluated:  
- **Baseline Models**: K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Logistic Regression  
- **Tree-Based Models**: Decision Tree, Random Forest  
- **Ensemble Models**: Gradient Boosting, XGBoost  
- **Bayesian Models**: Gaussian Naive Bayes  

Evaluation metrics include accuracy, precision, recall, F1-score, and Root Mean Squared Error (RMSE).  


---


## 🚀 Usage

### Requirements
- **Python 3.8+**  
- Libraries:  
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `streamlit`

### Steps to Run the Project
1. 🔗 Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/heart-disease-prediction.git
   cd heart-disease-prediction
   ```
2. 📦 Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. 📝 Run the Jupyter Notebook for data preprocessing and training:  
   ```bash
   jupyter notebook Project_Code.ipynb
   ```

---

## 🔮 Future Work

- 🌟 **Data Augmentation**: Address dataset imbalance for underrepresented classes.  
- 🛠️ **Hyperparameter Optimization**: Use grid search for fine-tuning models.  
- 🧠 **Explainability**: Integrate SHAP for model interpretability.  
- 📅 **Temporal Data Integration**: Include time-series data for dynamic risk prediction.  

---

## 🙌 Acknowledgments

Special thanks to:  
- 🏫 **Ramakrishna Mission Vivekananda Educational and Research Institute (RKMVERI)** for resources.  
- 👨‍🏫 **Dr. Tamal**, for guidance and support throughout the project.  

---
