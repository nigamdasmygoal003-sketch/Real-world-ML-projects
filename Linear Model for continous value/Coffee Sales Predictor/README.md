# ☕ Coffee Sales Prediction System

An end-to-end **Machine Learning regression project** that predicts coffee sales based on time, product, and contextual features.

This project demonstrates a **complete ML workflow** — from preprocessing and model training to deployment using a modern web interface.

---

## 🚀 Project Overview

The goal of this project is to predict the **sales amount (₹)** for a coffee shop transaction using features like:

* Time of day
* Coffee type
* Weekday
* Month
* Hour of purchase

The model is trained using **scikit-learn pipelines** and deployed using **Streamlit** for real-time predictions.

---

## 🧠 Problem Statement

> Predict the **continuous value of coffee sales** based on customer behavior and time-related patterns.

This helps businesses:

* Estimate expected revenue
* Understand peak sales hours
* Optimize staffing and inventory

---

## 🛠️ Tech Stack

* Python 🐍
* pandas, numpy
* scikit-learn
* joblib
* Streamlit

---

## 📁 Project Structure

```text
Coffee Sales Predictor/
│
├── data/
│   └── Coffe_sales.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model/
│   └── model.pkl
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd Coffee Sales Predictor

pip install -r requirements.txt
```

---

## 🧪 Model Training

```bash
python src/train.py
```

### What happens during training:

* Data loading & preprocessing
* Feature separation (numerical + categorical)
* Pipeline creation using:

  * `ColumnTransformer`
  * `Pipeline`
* Model training using **RandomForestRegressor**
* Evaluation using:

  * R² Score
  * RMSE
* Model saved to:

  ```
  model/model.pkl
  ```

---

## 📊 Model Performance

* **R² Score:** ~0.97
* **RMSE:** ~0.70

> Note: High performance is due to strong correlation between product type and price.

---

## 🔮 Prediction (CLI)

```bash
python src/predict.py
```

### Example Output:

```json
{
  "predicted_sales": 38.7
}
```

---

## 🌐 Run Web App

```bash
streamlit run streamlit_app.py
```

### Features:

* User-friendly input interface
* Real-time prediction
* Clean and responsive UI

---

## 🧠 Model Details

* Algorithm: **RandomForestRegressor**
* Preprocessing:

  * Missing value imputation
  * Standard scaling (numerical features)
  * One-hot encoding (categorical features)
* Pipeline ensures:

  * Reproducibility
  * Clean deployment

---

## 📌 Key Learnings

* Building end-to-end ML pipelines
* Handling mixed data types
* Deploying ML models with Streamlit
* Structuring production-ready ML projects
* Evaluating regression models using RMSE

---

## ⚠️ Limitations

* Model may behave like a **price lookup system** due to strong correlation between coffee type and price
* Does not account for:

  * Demand fluctuations
  * Seasonal trends
  * External factors (weather, promotions)

---

## 🚀 Future Improvements

* Convert to **revenue prediction system (aggregated data)**
* Add **time-series features (lag variables)**
* Implement **hyperparameter tuning (GridSearchCV)**
* Compare with advanced models (XGBoost, LightGBM)
* Deploy using **FastAPI / cloud platforms**
* Add monitoring & logging

---

## 🧑‍💻 Author

Nigam Das
B.Tech AI/ML Student | Machine Learning Enthusiast 🚀

---

## ⭐ Support

If you found this project useful:

* Give it a ⭐ on GitHub
* Share feedback

---

## 📢 Final Note

This project is part of a journey toward becoming a **production-ready Machine Learning Engineer**, focusing on real-world problem solving and deployment.
