# 🚗 Delhi Traffic Time Predictor

A Machine Learning project that predicts **travel time (in minutes)** for trips in Delhi based on traffic conditions, distance, weather, and other real-world factors.

Built using **Python, scikit-learn, and CustomTkinter** with a complete end-to-end pipeline from training to UI.

---

## 🚀 Features

* 📊 Predicts travel time using ML model
* 🧠 Model: Random Forest Regressor
* 🔄 Handles preprocessing:

  * MinMax Scaling (numerical features)
  * One-Hot Encoding (categorical features)
* 🖥️ Interactive desktop UI using **CustomTkinter**
* 📜 Scrollable UI for better usability
* 🎯 Styled output (bold + green prediction result)
* 💾 Model persistence using **joblib**

---

## 🧠 Machine Learning Workflow

1. Data Cleaning

   * Removed `Trip_ID`
2. Feature Processing

   * Numerical: Scaled using `MinMaxScaler`
   * Categorical: Encoded using `OneHotEncoder`
3. Train-Test Split

   * 70% Training / 30% Testing
4. Model Training

   * Random Forest Regressor
5. Evaluation Metrics

   * R² Score
   * RMSE

---

## 📊 Model Performance

* **R² Score:** ~0.99 ⚠️ *(May indicate possible data leakage — further validation recommended)*
* **RMSE:** ~2.9

---

## 📂 Project Structure

```bash
Delhi-Traffic-Predictor/
│
├── app.py                     # CustomTkinter GUI app
├── train.py                   # Model training script
├── scaler.pkl                 # Saved scaler
├── encoder.pkl                # Saved encoder
├── model.pkl                  # Trained model
├── delhi_traffic_features.csv
├── delhi_traffic_target.csv
├── requirements.txt
└── README.md
```

---

## 🖥️ How to Run

### 1. Clone the repository

Add **direct project link + instructions**

```md
## Project Folder
[Click here to open project](https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects/tree/main/Linear%20Model%20for%20continuous%20value/Delhi%20Traffic%20Time%20Predictor)
```
## Run Locally

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects.git
cd "Linear Model for continuous value/Delhi Traffic Time Predictor"
pip install -r requirements.txt
```
### 3. Run the application

```bash
python app.py
```

---

## 🧪 Input Features

| Feature               | Description                               |
| --------------------- | ----------------------------------------- |
| start_area            | Starting location                         |
| end_area              | Destination                               |
| distance_km           | Distance in kilometers                    |
| average_speed_kmph    | Average speed                             |
| time_of_day           | Time category (Morning Peak, Night, etc.) |
| day_of_week           | Weekday/Weekend                           |
| weather_condition     | Weather (Clear, Rain, Fog)                |
| traffic_density_level | Traffic level (Low/Medium/High)           |
| road_type             | Road type (Highway/Main/Inner)            |

---

## 🎯 Example Input

```text
Start Area: Vasant Kunj
End Area: Kalkaji
Distance: 9.44 km
Speed: 37.8 km/h
Time: Night
Day: Weekday
Weather: Clear
Traffic: Low
Road: Main Road
```

---

## 📌 Future Improvements

* 🔥 Convert to **Pipeline (sklearn)**
* 🌐 Deploy as **Web App (Streamlit/Flask)**
* ⚡ Hyperparameter tuning (GridSearchCV)
* 📉 Reduce data leakage & improve generalization
* 📊 Add visualization dashboard

---

## 💼 Resume Highlight

**Delhi Traffic Travel Time Prediction System**

* Built ML model achieving R² ≈ 0.99 (requires validation for leakage)
* Implemented preprocessing pipeline with scaling & encoding
* Designed interactive desktop UI using CustomTkinter
* Delivered end-to-end ML solution from data to deployment

---

## ⚠️ Note

The very high R² score may indicate **data leakage or overfitting**. Further validation using cross-validation is recommended.

---

## 🤝 Contributing

Feel free to fork this repository and improve it!

---

## ⭐ Acknowledgment

Built as part of real-world Machine Learning practice projects.
