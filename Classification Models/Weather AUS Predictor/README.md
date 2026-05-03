# 🌧️ Rain Prediction System (Weather Australia)

An end-to-end **Machine Learning classification project** that predicts whether it will rain tomorrow based on historical weather data.

Built using **Python, scikit-learn, and Streamlit**, this project demonstrates a complete ML workflow — from data preprocessing to deployment as a web application.

---

## 🚀 Features

* 📊 Data preprocessing using `Pipeline` & `ColumnTransformer`
* 🧠 Logistic Regression with class imbalance handling
* 📈 Model evaluation using Precision, Recall, F1-score
* ⚖️ Threshold tuning for better rain prediction (recall-focused)
* 💾 Model persistence using `joblib`
* 🌐 Interactive web app using Streamlit
* 🔮 Real-time prediction with probability & risk level

---

## 🧠 Problem Statement

Weather prediction is crucial for agriculture, travel, and daily planning.

Goal:

> Predict whether it will **rain tomorrow**, so users can make informed decisions.

---

## 📁 Project Structure

```text
Weather AUS Predictor/
│
├── data/
│   └── weatherAUS.csv
│
├── src/
│   └── train.py
│
├── model/
│   └── model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd Weather AUS Predictor

pip install -r requirements.txt
```

---

## 🧪 Model Training

```bash
python src/train.py
```

This will:

* Load and preprocess data
* Train the model using pipelines
* Evaluate performance
* Save the trained model to `model/model.pkl`

---

## 🌐 Run Web App

```bash
streamlit run app.py
```

---

## 🔮 Prediction Output

The app provides:

* Rain prediction (Yes / No)
* Probability of rain
* Risk level:

  * 🟢 Low Risk
  * 🟡 Medium Risk
  * 🔴 High Risk

---

## 🧠 Model Details

* Algorithm: **Logistic Regression**
* Class imbalance handled using:

  * `class_weight="balanced"`
* Preprocessing:

  * Missing value imputation (median & most frequent)
  * Standard scaling (numerical features)
  * One-hot encoding (categorical features)

---

## ⚖️ Business Logic

Instead of using default threshold (0.5):

* Custom threshold = **0.3**
* Reason:

  * Increase recall (catch more rainy days)
  * Reduce missed rain predictions

---

## 📊 Evaluation Metrics

* Accuracy: ~80%
* Recall (Rain class): ~78% ✅
* F1-score: Balanced performance

---

## 📌 Key Learnings

* Building end-to-end ML pipelines
* Handling missing data effectively
* Dealing with imbalanced classification
* Threshold tuning for real-world use
* Deploying ML models with Streamlit
* Solving training vs serving mismatch issues

---

## 🚀 Future Improvements

* Add advanced models (XGBoost, LightGBM)
* Feature importance visualization
* Model comparison dashboard
* Deploy on cloud (Render / AWS / HuggingFace)
* Add real-time weather API integration

---

## 🧑‍💻 Author

Nigam Das
B.Tech AI/ML Student | Machine Learning Enthusiast 🚀

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
