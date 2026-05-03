# 📊 KMeans Clustering 2D – Interactive ML Web App

An interactive **Machine Learning web application** that demonstrates **KMeans clustering** on 2D data, built using **scikit-learn** and deployed with **Streamlit**.

Users can input custom data points and instantly visualize which cluster they belong to.

---

## 🚀 Features

* 📊 KMeans Clustering using scikit-learn
* ⚙️ Pipeline-based preprocessing (StandardScaler + KMeans)
* 📉 Elbow Method for optimal cluster selection
* 🔮 Real-time cluster prediction
* 📍 Visualization of clusters, centroids, and user input
* 🌐 Interactive web app using Streamlit
* 💾 Model persistence using joblib

---

## 🧠 Problem Statement

Clustering is an **unsupervised learning technique** used to group similar data points.

This project demonstrates:

> How to automatically segment data into meaningful clusters **without labels**

---

## 📁 Project Structure

```text
KMeans Clustering 2D/
│
├── data/
│   └── dataset.csv
│
├── model/
│   └── kmeans_model.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app.py
├── streamlitapp.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd KMeans Clustering 2D

pip install -r requirements.txt
```

---

## 🏋️ Model Training

```bash
python src/train.py
```

This will:

* Load dataset
* Apply scaling
* Train KMeans model
* Save pipeline to `model/kmeans_model.pkl`

---

## 🔮 Prediction (CLI)

```bash
python src/predict.py
```

Example output:

```json
{
  "cluster": 1,
  "segment": "Cluster 2"
}
```

---

## 🌐 Run Web App

```bash
streamlit run app.py
```

---

## 🖥️ Web App Features

* Use sliders to input:

  * `x` value
  * `y` value
* View:

  * Predicted cluster
  * Cluster visualization
  * Centroids
  * Your input point highlighted

---

## 🧠 Model Details

* Algorithm: **KMeans Clustering**
* Number of clusters: **3**
* Preprocessing:

  * StandardScaler
* Pipeline:

  * Ensures consistent transformation during training and inference

---

## 📉 Elbow Method

Used to determine optimal number of clusters.

* WCSS (Within Cluster Sum of Squares) plotted
* Optimal cluster count chosen where curve bends

---

## 📊 Visualization

The app displays:

* Clustered data points
* Centroids (marked with "X")
* User input point (highlighted)

---

## 📌 Key Learnings

* Unsupervised learning workflow
* Importance of feature scaling in distance-based algorithms
* Pipeline usage for production-ready ML
* Model persistence using joblib
* Building interactive ML apps with Streamlit

---

## 🚀 Future Improvements

* Add DBSCAN clustering comparison
* Dynamic cluster selection (user-controlled)
* Deploy on Streamlit Cloud / Render
* Add 3D visualization
* Use real-world dataset (customer segmentation)

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

> Developed an interactive KMeans clustering web application using Streamlit, featuring real-time predictions, pipeline-based preprocessing, and dynamic visualization of clusters and centroids.
