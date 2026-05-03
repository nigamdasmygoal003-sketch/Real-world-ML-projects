# 🛍️ Mall Customer Segmentation (Unsupervised ML)

An end-to-end **Machine Learning project** that segments customers into meaningful groups using **KMeans Clustering**, built with **scikit-learn pipelines** and deployed as an interactive **Streamlit web app with PCA visualization**.

---

## 🚀 Features

* 🧠 KMeans Clustering using scikit-learn
* ⚙️ Pipeline (ColumnTransformer + StandardScaler + OneHotEncoder)
* 📊 Automatic preprocessing for numeric & categorical data
* 🔮 Real-time customer segment prediction
* 📉 Elbow Method for optimal cluster selection
* 🧬 PCA-based 2D visualization of clusters
* 🌐 Interactive Streamlit web app
* 💾 Model persistence using joblib

---

## 🧠 Problem Statement

Businesses need to understand different types of customers to improve marketing and retention.

Goal:

> Segment customers into groups based on behavior such as income and spending patterns.

---

## 📁 Project Structure

```text
Mall Customer Segmentation predictor/
│
├── data/
│   └── Mall_Customers.csv
│
├── notebooks/
│   └── experiment.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model/
│   └── kmeans_pipeline.pkl
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
[Click here to open project](https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects/tree/main/Unsupervised%20Learning%20Models/Mall%20Customer%20Segmentation%20predictor)
```
## Run Locally

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects.git
cd "Unsupervised Learning Models/Mall Customer Segmentation predictor"
pip install -r requirements.txt

```

## 🏋️ Model Training

```bash
python src/train.py
```

This will:

* Load dataset
* Apply preprocessing (scaling + encoding)
* Train KMeans clustering model
* Save pipeline to `model/kmeans_pipeline.pkl`
* Print cluster analysis

---

## 🔮 Prediction (CLI)

```bash
python src/predict.py
```

Example output:

```json
{
  "cluster": 2,
  "segment": "High Value Customers"
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

  * Gender
  * Age
  * Annual Income
  * Spending Score

* Output:

  * Predicted cluster
  * Customer segment

* Visualization:

  * PCA-based 2D cluster view
  * All customers plotted
  * User input highlighted

---

## 🧠 Model Details

* Algorithm: **KMeans Clustering**
* Number of clusters: **5**
* Preprocessing:

  * StandardScaler (numeric features)
  * OneHotEncoder (categorical features)
* Pipeline ensures:

  * Consistent transformation during training & inference

---

## 📊 Cluster Interpretation

Clusters are interpreted based on:

* Annual Income
* Spending Score
* Age

Example segments:

* 💎 High Income, High Spending → **VIP Customers**
* 💸 Low Income, Low Spending → **Budget Customers**
* 🎯 High Income, Low Spending → **Careful Customers**
* 🧑 Young, High Spending → **Target Customers**

---

## 📉 PCA Visualization

Principal Component Analysis (PCA) is used to:

* Reduce high-dimensional data → 2D
* Visualize clusters clearly
* Understand separation between customer groups

---

## 📌 Key Learnings

* Unsupervised learning workflow
* KMeans clustering in real-world use case
* Importance of preprocessing pipelines
* Handling categorical + numerical features
* Dimensionality reduction using PCA
* Building interactive ML apps with Streamlit

---

## 🚀 Future Improvements

* Add DBSCAN clustering comparison
* Allow dynamic cluster selection
* Deploy on Streamlit Cloud / Render
* Add interactive plots (Plotly)
* Integrate real-time data input

---

## 🧑‍💻 Author

Nigam Das
B.Tech AI/ML Student | Aspiring ML Engineer 🚀

---

## ⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork it
📢 Share it

---

## 💼 Resume Highlight

> Developed a customer segmentation system using KMeans with pipeline-based preprocessing and deployed an interactive Streamlit app with PCA visualization for real-time insights.
