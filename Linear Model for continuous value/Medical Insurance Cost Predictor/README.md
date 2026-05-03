# 💰 Medical Insurance Cost Predictor

A Machine Learning project that predicts **medical insurance charges** based on user details such as age, BMI, smoking status, and region.

Built using **Python, scikit-learn, and CustomTkinter**, this project provides an end-to-end solution from model training to an interactive desktop application.

---

## 🚀 Features

* 📊 Predicts insurance cost using ML model
* 🧠 Model: Random Forest Regressor
* 🔄 Preprocessing:

  * Standard Scaling (numerical features)
  * One-Hot Encoding (categorical features)
* 🖥️ Interactive GUI using **CustomTkinter**
* 📜 Scrollable UI for better user experience
* 🎯 Styled output (bold + green prediction result)
* 💾 Model saving using **joblib**

---

## 🧠 Machine Learning Workflow

1. Data Cleaning

   * Encoded `sex` and `smoker` columns
2. Feature Processing

   * Numerical features scaled using `StandardScaler`
   * Categorical feature (`region`) encoded using `OneHotEncoder`
3. Train-Test Split

   * 80% Training / 20% Testing
4. Model Training

   * Random Forest Regressor
5. Evaluation Metrics

   * R² Score
   * RMSE

---

## 📊 Model Performance

* **R² Score:** ~0.86 ✅
* **RMSE:** ~4600

👉 Indicates strong predictive performance without overfitting.

---

## 📂 Project Structure

```bash
Medical-Charges-Predictor/
│
├── app.py              # CustomTkinter GUI application
├── train.py            # Model training script
├── scaler.pkl          # Saved scaler
├── encoder.pkl         # Saved encoder
├── model.pkl           # Trained model
├── medical.csv         # Dataset
├── requirements.txt
└── README.md
```

---

## 🖥️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects
cd medical-charges-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

---

## 🧪 Input Features

| Feature  | Description        |
| -------- | ------------------ |
| age      | Age of the person  |
| bmi      | Body Mass Index    |
| children | Number of children |
| sex      | Male/Female        |
| smoker   | Yes/No             |
| region   | Residential region |

---

## 🎯 Example Input

```text
Age: 30
BMI: 28.5
Children: 2
Sex: Male
Smoker: No
Region: Southeast
```

---

## 📌 Future Improvements

* 🔥 Convert to **Pipeline (sklearn)**
* 🌐 Deploy as **Web App (Streamlit/Flask)**
* ⚡ Hyperparameter tuning (GridSearchCV)
* 📊 Add feature importance visualization
* 📉 Improve model performance (R² > 0.9)

---

## 💼 Resume Highlight

**Medical Insurance Cost Prediction System**

* Built ML model achieving **R² ≈ 0.86** using Random Forest
* Implemented preprocessing (scaling + encoding)
* Developed interactive desktop application using CustomTkinter
* Delivered complete end-to-end ML solution

---

## 🤝 Contributing

Feel free to fork and improve this project!

---

## ⭐ Acknowledgment

Built as part of real-world Machine Learning project practice.
