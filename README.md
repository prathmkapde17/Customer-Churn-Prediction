# Telco Customer Churn Prediction

---

## Overview

Customer churn prediction is a critical business problem in the telecom industry, where companies aim to identify customers who are likely to discontinue their services. Predicting churn accurately allows businesses to proactively engage at-risk customers, optimize retention strategies, and minimize revenue loss.

This project builds a machine learning pipeline to predict customer churn using the popular Telco Customer Churn dataset. It demonstrates data preprocessing, exploratory data analysis, feature engineering, handling imbalanced data, model training and tuning, and deploying the model with both Flask and FastAPI APIs.

---

## Problem Statement

Customer churn refers to customers who stop using a company's services within a certain period. Predicting churn early can help telecom providers offer targeted incentives and improve customer satisfaction, which directly impacts the company’s profitability.

The goal is to classify whether a customer will churn (`Yes`) or not (`No`) based on their demographic, service, and account information.

---

## Dataset

- Source: Publicly available Telco Customer Churn dataset.
- Features: Demographic info (gender, SeniorCitizen), account info (tenure, Contract), service subscriptions (InternetService, OnlineSecurity, StreamingTV), billing (MonthlyCharges, TotalCharges), and churn label.
- Target: `Churn` (Yes / No)

---

## Technology Stack

| Layer                | Tools & Libraries                             |
|----------------------|----------------------------------------------|
| Data Handling        | Python, Pandas, NumPy                         |
| Visualization        | Matplotlib, Seaborn                           |
| Data Preprocessing   | Scikit-learn, Imbalanced-learn (SMOTE)       |
| Modeling             | Scikit-learn (Random Forest), XGBoost        |
| Model Serialization  | Pickle                                        |
| Web Frameworks       | Flask (for web form), FastAPI (for API)      |
| API Testing          | Postman / Curl (for FastAPI endpoint testing)|
| Environment          | Python 3.7+, Jupyter Notebook/IDE             |

---

## Exploratory Data Analysis (EDA)

- Checked data shape and types.
- Dropped irrelevant columns (e.g., `customerID`).
- Identified categorical and numerical columns.
- Handled missing or inconsistent data (`TotalCharges` had empty strings replaced with 0.0).
- Visualized distributions of numerical features (tenure, MonthlyCharges, TotalCharges) using histograms and boxplots.
- Analyzed correlation between numerical features.
- Counted unique values and distributions of categorical variables.
- Explored churn rate distribution.

---

## Data Preprocessing

- **Encoding:** Label encoding of categorical columns using `LabelEncoder`.
- **Scaling:** Standardized numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) with `StandardScaler`.
- **Handling Imbalance:** The target variable (`Churn`) is imbalanced; applied SMOTE (Synthetic Minority Over-sampling Technique) on training data to balance classes.

---

## Modeling

Two models were trained and evaluated:

1. **Random Forest Classifier**  
2. **XGBoost Classifier**

- Both models were tuned using `GridSearchCV` with cross-validation to find the best hyperparameters.
- Random Forest showed the best balance of accuracy and interpretability.
- Saved the best model, encoders, and scalers using `pickle` for later use.

---

## Model Evaluation Metrics

- **Accuracy:** Overall correctness of predictions.
- **ROC-AUC Score:** Measures model’s ability to discriminate between classes.
- **Confusion Matrix:** Detailed breakdown of true/false positives/negatives.
- **Classification Report:** Precision, recall, F1-score for churn prediction.

Example model performance (Random Forest):

Accuracy : 0.79
ROC - AUC Score : 0.80

pgsql
Copy
Edit

---

## Deployment

### Flask Web App

- Simple web form for users to input customer data.
- On submission, the app predicts churn probability and label.
- Integrates model inference pipeline (encoding, scaling, prediction).
- Easy to run locally: `python app.py`

### FastAPI API

- REST API endpoint `/predict` accepts JSON payloads with customer features.
- Returns churn prediction and probability.
- Supports easy integration with other services or frontend apps.
- Run with: `uvicorn main:app --reload`

---

## Usage Example

```python
example_input = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

prediction, probability = make_prediction(example_input)
print(f"Prediction: {prediction}, Probability: {probability:.2f}")
Project Structure
graphql
Copy
Edit
├── app.py                 # Flask web app
├── main.py                # FastAPI API app
├── model_training.ipynb   # Notebook/script for EDA, training, and tuning
├── best_model.pkl         # Pickled Random Forest model
├── encoder.pkl            # Pickled LabelEncoders for categorical features
├── scaler.pkl             # Pickled StandardScaler for numerical features
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset file
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # HTML template for Flask form
└── README.md              # This documentation
Next Steps and Improvements
Add more advanced feature engineering (e.g., tenure buckets, interaction terms).

Experiment with deep learning models or ensemble stacking.

Implement real-time prediction with frontend integration.

Add input validation and error handling in web apps.

Deploy models on cloud platforms like AWS, GCP, or Azure for scalability.

Track experiments and model versions with tools like MLflow or DVC.

