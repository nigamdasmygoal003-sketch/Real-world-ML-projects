# 📊 DBSCAN Clustering & Outlier Detection System

An end-to-end **Unsupervised Machine Learning project** that identifies natural clusters and detects anomalies in data using **DBSCAN (Density-Based Spatial Clustering)**, built with **scikit-learn pipelines** and deployed as an interactive **Streamlit web app**.

---

## 🚀 Features

* 🧠 Density-based clustering using **DBSCAN**
* ⚙️ Pipeline-based preprocessing (StandardScaler)
* 🔍 Automatic detection of **outliers (noise points)**
* 📊 Visualization of clusters and anomalies
* 🔮 Real-time anomaly detection via Streamlit UI
* 💾 Model persistence using joblib

---

## 🧠 Problem Statement

Traditional clustering algorithms (like KMeans) struggle with:

* Irregular cluster shapes
* Noise and outliers

Goal:

> Identify natural clusters and detect abnormal data points using a density-based approach.

---

## 📁 Project Structure

```text id="yz8y68"
DBSCAN Clustering & Outlier Detection System/
│
├── data/
│   └── DBSCAN.csv
│
├── notebooks/
│   └── experiment.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model/
│   └── dbscan_pipeline.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Add **direct project link + instructions**

```md
## Project Folder
[Click here to open project](https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects/tree/main/Unsupervised%20Learning%20Models/DBSCAN%20Clustering%20%26%20Outlier%20Detection%20System)
```
## Run Locally

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects.git
cd "Unsupervised Learning Models/DBSCAN Clustering & Outlier Detection System"
pip install -r requirements.txt

```

## 🏋️ Model Training

```bash id="p9bxxv"
python src/train.py
```

This will:

* Load dataset
* Apply scaling
* Train DBSCAN model
* Identify clusters and noise points
* Save pipeline to `model/dbscan_pipeline.pkl`

---

## 🔮 Prediction (Important Note)

⚠️ DBSCAN does **NOT support `.predict()`**

Instead:

* New data points are evaluated using **distance-based logic**
* Compared with existing cluster density

---

## 🌐 Run Web App

```bash id="zcxqmo"
streamlit run app.py
```

---

## 🖥️ Web App Features

* Input:

  * Weight
  * Height

* Output:

  * Normal data point OR Outlier detection
  * Distance from nearest cluster
  * Comparison with `eps` threshold

* Visualization:

  * Clustered data points
  * Outliers highlighted
  * User input point marked

---

## 🧠 Model Details

* Algorithm: **DBSCAN**
* Parameters:

  * `eps = 0.2`
  * `min_samples = 5`
* Preprocessing:

  * StandardScaler (for distance normalization)

---

## 📊 How DBSCAN Works

* Groups points based on density
* Requires:

  * Minimum number of points (`min_samples`)
  * Maximum distance (`eps`)
* Labels:

  * `0,1,2...` → clusters
  * `-1` → noise (outliers)

---

## 📌 Key Learnings

* Difference between KMeans and DBSCAN
* Handling irregular clusters
* Detecting anomalies in data
* Importance of scaling in distance-based models
* Limitations of DBSCAN (no `.predict()`)

---

## 🚀 Future Improvements

* Add interactive visualization (Plotly)
* Dynamic tuning of `eps` parameter
* Compare with Isolation Forest
* Extend to real-world anomaly detection datasets
* Deploy on Streamlit Cloud / Render

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

> Built an anomaly detection system using DBSCAN to identify clusters and outliers, with an interactive Streamlit app for real-time visualization and analysis.
