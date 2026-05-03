# 📊 Customer Churn Prediction System

An end-to-end **Machine Learning project** that predicts whether a customer will churn (leave the service) based on behavioral and service-related features.

Built using **Python, scikit-learn, and CustomTkinter**, this project demonstrates a full ML workflow from data preprocessing to deployment as a desktop application.

---

## 🚀 Features

* 📊 Data preprocessing using `Pipeline` & `ColumnTransformer`
* 🧠 Logistic Regression with class imbalance handling
* 📈 Model evaluation using F1-score and classification report
* ⚖️ Threshold tuning for business decision-making
* 💾 Model persistence using `joblib`
* 🔮 Prediction system with churn probability
* 🖥️ Desktop UI built with CustomTkinter

---

## 🧠 Problem Statement

Customer churn is a major problem for telecom companies.

Goal:

> Predict whether a customer will **leave (churn)** or **stay**, so the company can take preventive action.

---

## 📁 Project Structure

```text
Customer Churn Predictor/
│
├── data/
│   └── customer_churn.csv
│
├── notebooks/
│   └── CustomerChurnPredictor.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model/
│   └── churn_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects
cd Customer Churn Predictor

pip install -r requirements.txt
```

---

## 🧪 Model Training

```bash
python src/train.py
```

This will:

* Clean and preprocess data
* Train the model
* Save it to `model/churn_model.pkl`

---

## 🔮 Prediction (CLI)

```bash
python src/predict.py
```

Example output:

```json
{
  "churn": true,
  "churn_probability": 0.67,
  "threshold_used": 0.4
}
```

---

## 🖥️ Run Application

```bash
python app.py
```

A desktop application will open where users can:

* Input customer details
* Click **Predict**
* View churn risk and probability

---

## 🧠 Model Details

* Algorithm: **Logistic Regression**
* Class imbalance handled using:

  * `class_weight="balanced"`
* Preprocessing:

  * Missing value imputation
  * Standard scaling
  * One-hot encoding
* Feature Engineering:

  * `avg_monthly_spend`

---

## ⚖️ Business Logic

* Model outputs churn probability
* Custom threshold (default = 0.4) used to:

  * Increase recall for churn customers
  * Reduce missed churn cases

---

## 📊 Evaluation Metrics

* Accuracy
* F1-score
* Recall (focused on churn class)

---

## 📌 Key Learnings

* Handling imbalanced datasets
* Building ML pipelines (ColumnTransformer + Pipeline)
* Model evaluation beyond accuracy
* Threshold tuning for business decisions
* Building deployable ML applications

---

## 🚀 Future Improvements

* Convert to Streamlit web app
* Deploy using FastAPI / Flask
* Add model monitoring & logging
* Improve UI with better styling & grouping
* Use advanced models (XGBoost, LightGBM)

---

## 🧑‍💻 Author

Nigam Das
Machine Learning Enthusiast 🚀

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
