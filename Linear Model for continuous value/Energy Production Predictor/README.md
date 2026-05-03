# ⚡ Energy Production Predictor

A Machine Learning project that predicts energy production based on time and environmental factors using **scikit-learn** and a **CustomTkinter desktop application**.

---

## 🚀 Features

* 📊 Predicts energy production using trained ML model
* 🧠 Uses **Random Forest Regressor**
* 🔄 Handles preprocessing (Scaling + Encoding)
* 🖥️ Interactive desktop UI built with **CustomTkinter**
* 💾 Model persistence using **joblib**
* 📁 Clean and modular project structure

---

## 🧠 Machine Learning Workflow

1. Data Preprocessing

   * Numerical scaling using `StandardScaler`
   * Categorical encoding using `OneHotEncoder`

2. Model Training

   * Model: Random Forest Regressor
   * Tuned parameters:

     * `n_estimators=200`
     * `max_depth=50`
     * `min_samples_split=5`
     * `min_samples_leaf=2`

3. Evaluation Metrics

   * R² Score: **0.55+**
   * RMSE: **~2655**

---

## 📂 Project Structure

```
Energy-Production-Predictor/
│
├── app.py               # CustomTkinter UI app
├── train.py             # Model training script
├── scaler.pkl           # Saved scaler
├── encoder.pkl          # Saved encoder
├── model.pkl            # Trained model
├── EnergyProductionDataset.csv
├── requirements.txt
└── README.md
```

---

## 🖥️ How to Run

### 1. Clone the repository

Add **direct project link + instructions**

```md
## Project Folder
[Click here to open project](https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects/tree/main/Linear%20Model%20for%20continuous%20value/Energy%20Production%20Predictor)
```
## Run Locally

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects.git
cd "Linear Model for continuous value/Energy Production Predictor"
pip install -r requirements.txt
```
### 3. Run the app

```
python app.py
```

---

## 🧪 Input Features

| Feature     | Description                        |
| ----------- | ---------------------------------- |
| Start_Hour  | Hour when energy generation starts |
| End_Hour    | Hour when energy generation ends   |
| Source      | Energy source (Solar/Wind)         |
| Day_of_Year | Day number in the year             |
| Day_Name    | Day of the week                    |
| Month_Name  | Month name                         |
| Season      | Season (Winter/Summer/etc.)        |

---

## 🎯 Example Input

```
Start Hour: 7
End Hour: 8
Source: Wind
Day of Year: 356
Day Name: Friday
Month: December
Season: Winter
```

---

## 📌 Future Improvements

* 🔥 Convert to **Pipeline (sklearn)**
* 🌐 Deploy as **web app (Streamlit/Flask)**
* 📈 Add feature engineering (Duration, cyclical encoding)
* ⚡ Hyperparameter tuning with GridSearchCV

---

## 💼 Resume Highlight

**Energy Production Prediction System**

* Built an ML model using Random Forest achieving R² > 0.55
* Designed a desktop application using CustomTkinter
* Implemented preprocessing pipelines and model persistence
* Delivered an end-to-end ML solution from training to UI

---

## 🤝 Contributing

Feel free to fork this repo and improve it!

---

## ⭐ Acknowledgment

Built as part of hands-on Machine Learning project practice.
