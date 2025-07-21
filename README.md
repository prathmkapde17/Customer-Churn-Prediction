# 📉 Customer Churn Prediction 🚀

Predict whether a customer will churn using machine learning techniques! This project leverages a real-world telecom dataset to develop, train, and deploy a churn prediction model with both Flask and FastAPI for interactive inference.

---

## 📦 Project Features

- 📊 **Exploratory Data Analysis (EDA):** Visual insights to understand customer behavior.
- 🧹 **Data Preprocessing:** Encoding categorical variables, feature scaling, and handling missing values.
- 🤖 **Modeling:** Trained multiple models, with the best-performing.
- 🌐 **Web Deployment:**
  - `app.py` for a **Flask** web interface.
  - `fastapi_app.py` for a **FastAPI** REST API.
- ✅ **Prediction Outputs:** Whether the customer is likely to churn, and associated probability.

---

## 🛠 Tech Stack

- **Python** 🐍
- **Pandas, Scikit-learn, Pickle** for data manipulation & ML
- **Flask** 🌐 (frontend deployment)
- **FastAPI** ⚡ (API deployment)

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Flask App

```bash
python app.py
```

Navigate to: `http://localhost:5000`

### 3. Run FastAPI App

```bash
uvicorn fastapi_app:app --reload
```

Access API docs at: `http://127.0.0.1:8000/docs`

---

## 📁 Dataset

Used the **Telco Customer Churn** dataset:  
`WA_Fn-UseC_-Telco-Customer-Churn.csv`  
Includes features like tenure, internet services, contract type, and billing info.

---

## 🧠 Prediction Logic

Inputs are preprocessed (encoded + scaled), passed into the ML model, and results include:

- 🎯 **Prediction:** "Churn" or "No Churn"
- 📈 **Probability Score**

---

## 🙌 Acknowledgments

Thanks to IBM for the open-source dataset.  
Built with ❤️ by [prathmkapde17].
