# 🧠 Mental Health & Digital Behavior Segmentation

An end-to-end **Machine Learning project** that analyzes digital behavior patterns and segments users based on their **mental wellbeing**, built using **KMeans clustering**, **scikit-learn pipelines**, and deployed with an interactive **Streamlit web app with PCA visualization**.

---

## 🚀 Features

* 🧠 Unsupervised Learning using **KMeans Clustering**
* ⚙️ Pipeline-based preprocessing (StandardScaler)
* 🔮 Real-time user segmentation prediction
* 📊 PCA-based visualization of behavioral clusters
* 🧾 Insight generation based on user behavior
* 🌐 Interactive web application using Streamlit
* 💾 Model persistence using joblib

---

## 🧠 Problem Statement

Modern digital habits (screen time, notifications, social media usage) strongly impact mental health.

Goal:

> Segment users into meaningful groups based on their digital behavior and mental health indicators.

---

## 📁 Project Structure

```text
Mental Health and Digital Behavior predictor/
│
├── data/
│   └── mental_health_digital_behavior_data.csv
│
├── notebooks/
│   └── experiment.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
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

Add **direct project link + instructions**

```md
## Project Folder
[Click here to open project](https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects/tree/main/Unsupervised%20Learning%20Models/Mental%20Health%20and%20Digital%20Behavior%20predictor)
```
## Run Locally

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects.git
cd "Unsupervised Learning Models/Mental Health and Digital Behavior predictor"
pip install -r requirements.txt

```

## 🏋️ Model Training

```bash
python src/train.py
```

This will:

* Load dataset
* Apply scaling using pipeline
* Train KMeans clustering model
* Perform cluster analysis
* Save model to `model/model.pkl`

---

## 🔮 Prediction (CLI)

```bash
python src/predict.py
```

Example output:

```json
{
  "cluster": 1,
  "segment": "Digitally Overloaded Users"
}
```

---

## 🌐 Run Web App

```bash
streamlit run app.py
```

---

## 🖥️ Web App Features

* Sidebar inputs:

  * Screen time
  * Sleep hours
  * Notifications
  * Social media usage
  * Mood & anxiety scores

* Output:

  * User segment
  * Behavioral insight

* Visualization:

  * PCA-based 2D cluster view
  * User position highlighted

---

## 🧠 Model Details

* Algorithm: **KMeans Clustering**
* Number of clusters: **3**
* Preprocessing:

  * StandardScaler (all features)
* Pipeline ensures:

  * Consistent transformations during training and inference

---

## 📊 Cluster Interpretation

The model identifies three key user groups:

### 🟩 Healthy Balanced Users

* Good sleep
* Low anxiety
* High wellbeing
  👉 Healthy digital habits

---

### 🟥 Digitally Overloaded Users

* High screen time
* Low sleep
* High anxiety
  👉 High-risk group

---

### 🟨 At-Risk Users

* Moderate usage
* Elevated anxiety
  👉 Hidden risk group

---

## 📉 PCA Visualization

Principal Component Analysis (PCA) is used to:

* Reduce multi-dimensional data → 2D
* Visualize cluster separation
* Understand behavioral patterns

---

## 📌 Key Learnings

* Unsupervised learning in real-world scenarios
* Behavioral data clustering
* Importance of preprocessing pipelines
* Dimensionality reduction using PCA
* Building interactive ML applications with Streamlit

---

## 🚀 Future Improvements

* Add recommendation system (personalized advice)
* Compare with DBSCAN for anomaly detection
* Deploy on Streamlit Cloud / Render
* Add real-time data tracking
* Use advanced models (Isolation Forest)

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

> Built a mental health segmentation system using KMeans clustering with pipeline-based preprocessing and deployed an interactive Streamlit app with PCA visualization and behavioral insights.
