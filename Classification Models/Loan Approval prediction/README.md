# 💼 Loan Approval Prediction System

A complete **Machine Learning project** that predicts whether a loan should be approved or rejected based on applicant details.
Built using **Python, scikit-learn, and CustomTkinter** with an end-to-end pipeline from data analysis to deployment.

---

## 🚀 Features

* 📊 Data preprocessing using `Pipeline` & `ColumnTransformer`
* 🤖 Model training with multiple classifiers
* 📈 Model evaluation using F1-score & cross-validation
* 💾 Model saving using `joblib`
* 🔮 Prediction system with probability output
* 🖥️ Desktop application using CustomTkinter

---

## 🧠 Problem Statement

Banks need to decide whether a loan application should be approved based on applicant details such as:

* Income
* Credit Score
* Loan Amount
* Employment History
* Location

This project builds a machine learning model to automate that decision.

---

## 🏗️ Project Structure

```
Loan Approval prediction/
│
├── data/
│   └── loan_approval.csv
│
├── notebooks/
│   └── EDA_Model_Training.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model/
│   └── loan_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects
cd Loan Approval prediction

pip install -r requirements.txt
```

---

## 🧪 Model Training

```bash
python src/train.py
```

This will:

* Train the model using full dataset
* Save it to `model/loan_model.pkl`

---

## 🔮 Prediction (CLI)

```bash
python src/predict.py
```

Example output:

```
{'prediction': True, 'approval_probability': 0.92}
```

---

## 🖥️ Run Application

```bash
python app.py
```

A desktop UI will open where you can:

* Enter applicant details
* Click **Predict**
* Get approval decision + confidence score

---

## 📊 Model Details

* Algorithm: **RandomForestClassifier**
* Preprocessing:

  * Missing value handling
  * Feature scaling
  * One-hot encoding
* Evaluation Metrics:

  * Accuracy
  * F1-score
  * Cross-validation

---

## ⚖️ Business Logic

* Model predicts probability of loan approval
* Custom threshold used to control risk
* Helps reduce **false approvals (risky customers)**

---

## 📌 Key Learnings

* End-to-end ML pipeline development
* Model comparison & selection
* Feature preprocessing using pipelines
* Building production-ready ML scripts
* Creating desktop ML applications

---

## 🚀 Future Improvements

* Add Streamlit web deployment
* Add API using Flask/FastAPI
* Improve UI with dropdowns & validation
* Use real-world dataset for better generalization
* Add model monitoring & logging

---

## 🧑‍💻 Author

Nigam Das
Machine Learning Enthusiast 🚀

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
