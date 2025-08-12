# 📊 Telecom Customer Churn Prediction

## 📌 Overview
Customer churn is a critical challenge in the telecom industry — identifying customers likely to discontinue services allows companies to take preventive actions, improve retention, and minimize revenue loss.

This **end-to-end machine learning project** predicts churn using the **Telco Customer Churn dataset**. It covers the complete ML pipeline:  
✔ Data Cleaning & Preprocessing  
✔ Exploratory Data Analysis (EDA)  
✔ Feature Encoding & Scaling  
✔ Handling Class Imbalance with SMOTE  
✔ Model Training & Hyperparameter Tuning (Random Forest & XGBoost)  
✔ Evaluation with multiple metrics  
✔ Deployment as both **Flask Web App** and **FastAPI API**

---

## 📂 Dataset
- **Source:** IBM Sample Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Features Include:**
  - **Demographics:** Gender, SeniorCitizen, Partner, Dependents
  - **Account Info:** Tenure, Contract type, Payment method
  - **Services:** Phone service, Internet service, Streaming services
  - **Billing:** MonthlyCharges, TotalCharges
- **Target Variable:** `Churn` → Yes / No

---

## 🛠 Technology Stack
**Data & Modeling:**
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn, XGBoost
- imbalanced-learn (SMOTE)

**Deployment:**
- Flask (Web Form)
- FastAPI (REST API)
- Pickle for model serialization

---

## 📈 Workflow

### 1. **Data Preprocessing**
- Dropped `customerID`
- Replaced blank `TotalCharges` with `0.0` and converted to float
- Label encoded categorical features
- Scaled numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) using StandardScaler

### 2. **EDA**
- Histograms & Boxplots for numeric variables
- Count plots for categorical features
- Correlation heatmap

### 3. **Balancing Data**
- Applied **SMOTE** to handle imbalanced churn labels

### 4. **Model Training**
- **Random Forest Classifier** & **XGBClassifier**
- Tuned hyperparameters with **GridSearchCV**
- Selected best model (Random Forest) for deployment

### 5. **Evaluation Metrics**
- Accuracy Score
- ROC-AUC
- Confusion Matrix
- Classification Report

### 6. **Deployment**
- **Flask App:** Input form for customer details → shows churn prediction & probability  
- **FastAPI App:** `/predict` endpoint → accepts JSON → responds with prediction & probability

---

## 📂 Project Structure

📁 Telecom-Customer-Churn-Prediction
│-- WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
│-- model_training.py # Data preprocessing + model training
│-- app_flask.py # Flask web app
│-- app_fastapi.py # FastAPI service
│-- templates/
│ └── index.html # Flask web form template
│-- best_model.pkl # Saved churn prediction model
│-- encoder.pkl # LabelEncoders for categorical data
│-- scaler.pkl # StandardScaler for numeric data
│-- requirements.txt # Project dependencies
│-- README.md # Project documentation

text

---

## 🚀 How to Run

### 1️⃣ Clone Repository
git clone https://github.com/prathmkapde17/Telecom-Customer-Churn-Prediction.git
cd Telecom-Customer-Churn-Prediction

text

### 2️⃣ Install Dependencies
pip install -r requirements.txt

text

### 3️⃣ (Optional) Retrain Model
python model_training.py

text

### 4️⃣ Run Flask Web App
python app_flask.py

text
Go to: [**http://127.0.0.1:5000/**](http://127.0.0.1:5000/)

### 5️⃣ Run FastAPI App
uvicorn app_fastapi:app --reload

text
API Endpoint: **POST** → `http://127.0.0.1:8000/predict`

---

## 📌 Example API Request
{
"gender": "Female",
"SeniorCitizen": 0,
"Partner": "Yes",
"Dependents": "No",
"tenure": 1,
"PhoneService": "No",
"MultipleLines": "No phone service",
"InternetService": "DSL",
"OnlineSecurity": "No",
"OnlineBackup": "Yes",
"DeviceProtection": "No",
"TechSupport": "No",
"StreamingTV": "No",
"StreamingMovies": "No",
"Contract": "Month-to-month",
"PaperlessBilling": "Yes",
"PaymentMethod": "Electronic check",
"MonthlyCharges": 29.85,
"TotalCharges": 29.85
}

text

---

## 📊 Sample Output
Prediction: Churn
Probability: 0.78

text

---

## 🔮 Future Improvements
- Add feature importance analysis using SHAP/LIME
- Deploy to cloud (AWS/GCP/Azure)
- Add data validation & logging for production
- Containerize with Docker

---

## 👤 Author
**Prathmesh Kapde**  
📧 prathmkapde17@gmail.com  
💼 Machine Learning & Data Science Enthusiast
