# 🏦 Bank Transaction Fraud Detection (Unsupervised ML)

An end-to-end **Machine Learning project** that detects suspicious banking transactions using **Isolation Forest (Anomaly Detection)**, built with **scikit-learn pipelines** and deployed as an interactive **Streamlit web application**.

---

## 🚀 Features

* 🧠 Anomaly Detection using **Isolation Forest**
* ⚙️ Pipeline-based preprocessing (StandardScaler + OneHotEncoder)
* 🔍 Detection of unusual transaction patterns
* 🔮 Real-time fraud prediction via Streamlit UI
* 📊 Behavioral risk analysis (rule-based insights)
* 💾 Model persistence using joblib

---

## 🧠 Problem Statement

In modern banking systems, fraud detection is critical.

However:

* Most datasets do **not have labeled fraud data**
* Traditional supervised models cannot be applied

Goal:

> Detect suspicious transactions by identifying patterns that deviate from normal behavior using unsupervised learning.

---

## 📁 Project Structure

```text id="fraud-structure"
Bank Transactions Dataset for Fraud Detection/
│
├── data/
│   └── bank_transactions.csv
│
├── notebooks/
│   └── experiment.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model/
│   └── isolation_forest_pipeline.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash id="fraud-install"
git clone <your-repo-url>
cd Bank Transactions Dataset for Fraud Detection

pip install -r requirements.txt
```

---

## 🏋️ Model Training

```bash id="fraud-train"
python src/train.py
```

This will:

* Load dataset
* Clean unnecessary columns (IDs, IP, etc.)
* Apply preprocessing pipeline
* Train Isolation Forest model
* Detect anomalies
* Save model to `model/isolation_forest_pipeline.pkl`

---

## 🔮 Prediction (CLI)

```bash id="fraud-predict"
python src/predict.py
```

Example output:

```json id="fraud-output"
{
  "prediction": "Anomaly",
  "status": "Suspicious Transaction"
}
```

---

## 🌐 Run Web App

```bash id="fraud-run"
streamlit run app.py
```

---

## 🖥️ Web App Features

* Input transaction details:

  * Transaction Amount
  * Duration
  * Login Attempts
  * Account Balance
  * Customer Age
  * Transaction Type
  * Channel
  * Location
  * Occupation

* Output:

  * Normal or Suspicious transaction
  * Risk insights based on behavior

---

## 🧠 Model Details

* Algorithm: **Isolation Forest**
* Type: Unsupervised anomaly detection
* Contamination: ~2% anomalies
* Preprocessing:

  * StandardScaler (numerical features)
  * OneHotEncoder (categorical features)

---

## 📊 How Isolation Forest Works

* Randomly isolates data points
* Anomalies require **fewer splits to isolate**
* Works efficiently on high-dimensional data

Output:

* `1` → Normal
* `-1` → Anomaly

---

## 📌 Key Learnings

* Unsupervised anomaly detection
* Handling mixed data (categorical + numerical)
* Importance of pipelines in production ML
* Feature selection for fraud detection
* Building deployable ML systems

---

## 🚀 Future Improvements

* Add anomaly score visualization
* Integrate real-time streaming data
* Compare with DBSCAN and LOF
* Deploy using FastAPI backend
* Add batch prediction (CSV upload)

---

## 🧑‍💻 Author

Nigam Das
B.Tech AI/ML Student | Aspiring ML Engineer 🚀

---

## ⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork it
📢 Share it

---

## 💼 Resume Highlight

> Developed a fraud detection system using Isolation Forest with pipeline-based preprocessing and deployed an interactive Streamlit application for real-time anomaly detection and risk analysis.
