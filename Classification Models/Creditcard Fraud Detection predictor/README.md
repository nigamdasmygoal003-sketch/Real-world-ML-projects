# 🚨 Credit Card Fraud Detection System

An advanced **Machine Learning project** designed to detect fraudulent credit card transactions using highly imbalanced data.

This project demonstrates **real-world ML engineering techniques** including imbalance handling, SMOTE, threshold tuning, and deployment using a desktop application.

---

## 🚀 Features

* ⚖️ Handles extreme class imbalance (~0.17% fraud)
* 🧠 Uses **RandomForestClassifier** with SMOTE
* 📊 Data preprocessing with Pipeline
* 🔥 Threshold tuning for better fraud detection
* 📈 Evaluation using Recall, Precision, F1-score
* 💾 Model saving using `joblib`
* 🖥️ Desktop UI using CustomTkinter
* 📊 Data visualization using Plotly

---

## 🧠 Problem Statement

Fraud detection is a critical problem in banking systems.

Goal:

> Identify fraudulent transactions while minimizing missed fraud cases.

---

## 📊 Dataset

* Source: Credit Card Fraud Detection dataset
* Total samples: **284,807**
* Fraud cases: **492 (~0.17%)**

---

## ⚠️ Key Challenge

Extreme class imbalance:

| Class      | Count   |
| ---------- | ------- |
| Normal (0) | 284,315 |
| Fraud (1)  | 492     |

👉 Accuracy is misleading → focus on Recall & F1-score

---

## 🏗️ Project Structure

```text
Creditcard Fraud Detection predictor/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── EDA_creditcard_fraud_detection.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model/
│   └── fraud_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects
cd Creditcard Fraud Detection predictor

pip install -r requirements.txt
```

---

## 🧪 Model Training

```bash
python src/train.py
```

This will:

* Apply SMOTE to balance data
* Train RandomForest model
* Save model to `model/fraud_model.pkl`

---

## 🔮 Prediction (CLI)

```bash
python src/predict.py
```

Example output:

```json
{
  "fraud": true,
  "fraud_probability": 0.8123,
  "threshold_used": 0.3
}
```

---

## 🖥️ Run Application

```bash
python app.py
```

Features:

* Quick input mode (few features)
* JSON input mode (full transaction)
* Real-time fraud prediction

---

## 🧠 Model Details

* Algorithm: **RandomForestClassifier**
* Parameters:

  * `n_estimators=200`
  * `class_weight="balanced"`
* Imbalance Handling:

  * SMOTE (Synthetic Minority Oversampling)
* Preprocessing:

  * StandardScaler
* Evaluation Metrics:

  * Recall (Fraud class)
  * Precision
  * F1-score

---

## ⚖️ Business Logic

* Model outputs probability of fraud
* Custom threshold (default = 0.3)
* Lower threshold → catch more fraud
* Trade-off between:

  * False positives
  * Missed fraud cases

---

## 📊 Visualizations

* Class imbalance distribution
* Transaction amount vs fraud
* Time-based fraud patterns
* Feature correlation heatmap

---

## 📌 Key Learnings

* Handling imbalanced datasets (SMOTE)
* Building ML pipelines (ImbPipeline)
* Threshold tuning for real-world systems
* Evaluating models beyond accuracy
* Designing ML applications (UI + backend)

---

## 🚀 Future Improvements

* Deploy using Streamlit (web app)
* Add real-time API (FastAPI)
* Use advanced models (XGBoost, LightGBM)
* Add model monitoring & logging
* Improve UI/UX

---

## 🧑‍💻 Author

Nigam Das
Machine Learning Enthusiast 🚀

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
