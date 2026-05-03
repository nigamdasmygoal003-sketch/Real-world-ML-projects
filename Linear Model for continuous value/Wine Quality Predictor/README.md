# 🍷 Wine Quality Predictor

An end-to-end Machine Learning project that predicts whether a wine is **Good** or **Bad** based on its physicochemical properties. This project demonstrates data preprocessing, model training, evaluation, and deployment through an interactive desktop application built using **CustomTkinter**.

---

## 🚀 Project Overview

Wine quality is influenced by multiple chemical attributes such as acidity, sugar content, alcohol level, and pH. This project uses these features to classify wine into:

* ✅ **Good Quality (1)** — Quality ≥ 6
* ❌ **Bad Quality (0)** — Quality < 6

The goal is to build a robust ML system that can assist in quick quality assessment.

---

## 🧠 Machine Learning Workflow

### 1. Data Loading

* Dataset: `WineQuality.csv`
* Target: `quality`

### 2. Problem Transformation

* Converted regression → binary classification:

```python
y = (data["quality"] >= 6).astype(int)
```

### 3. Train-Test Split

* 80% training / 20% testing

### 4. Model Selection

* Model used: **Random Forest Classifier**
* Reason:

  * Handles non-linear relationships
  * Works well with tabular data
  * Robust to noise

### 5. Model Training

* Trained on 11 chemical features

### 6. Evaluation

* Metric used: **Accuracy**

---

## 📊 Model Performance

* **Accuracy:** ~80%–85% ✅
* Balanced performance across classes
* Good generalization on unseen data

---

## 📂 Project Structure

```bash
Wine-Quality-Predictor/
│
├── app.py                # CustomTkinter GUI application
├── train.py              # Model training script
├── model.pkl             # Trained model
├── WineQuality.csv       # Dataset
├── requirements.txt
└── README.md
```

---

## 🧪 Features Used

| Feature              | Description                        |
| -------------------- | ---------------------------------- |
| fixed acidity        | Acidity level of wine              |
| volatile acidity     | Amount of acetic acid              |
| citric acid          | Freshness indicator                |
| residual sugar       | Sugar remaining after fermentation |
| chlorides            | Salt content                       |
| free sulfur dioxide  | Prevents microbial growth          |
| total sulfur dioxide | Total SO₂ level                    |
| density              | Density of wine                    |
| pH                   | Acidity/alkalinity                 |
| sulphates            | Preservative                       |
| alcohol              | Alcohol percentage                 |

---

## 🖥️ Application (CustomTkinter UI)

### Features:

* Scrollable input form
* Input validation
* Predict button
* Color-coded output:

  * 🟢 Green → Good wine
  * 🔴 Red → Bad wine

---

## ▶️ How to Run

### 1. Clone the repository

Add **direct project link + instructions**

```md
## Project Folder
[Click here to open project](https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects/tree/main/Linear%20Model%20for%20continuous%20value/Wine%20Quality%20Predictor)
```
## Run Locally

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects.git
cd "Linear Model for continuous value/Wine Quality Predictor"
pip install -r requirements.txt
```
### 3. Run the application

```bash
python app.py
```

---

## 🎯 Example Input

```text
fixed acidity: 7.4
volatile acidity: 0.70
citric acid: 0.00
residual sugar: 1.9
chlorides: 0.076
free sulfur dioxide: 11
total sulfur dioxide: 34
density: 0.9978
pH: 3.51
sulphates: 0.56
alcohol: 9.4
```

### Output:

```
🍷 Good Quality Wine
```

---

## 📌 Key Learnings

* Understanding when to use **classification vs regression**
* Importance of **problem framing**
* Building **end-to-end ML systems**
* Integrating ML models with **GUI applications**
* Handling real-world noisy datasets

---

## 🔥 Future Improvements

* Add probability/confidence score
* Add feature importance visualization
* Convert to **Streamlit web app**
* Deploy online (Render / HuggingFace Spaces)

---

## 💼 Resume Highlight

**Wine Quality Classification System**

* Built ML model achieving ~85% accuracy using Random Forest
* Converted regression problem into classification for better performance
* Designed interactive desktop UI using CustomTkinter
* Developed complete ML pipeline from data preprocessing to deployment

---

## 🤝 Contributing

Feel free to fork the repository and enhance it!

---

## ⭐ Acknowledgment

This project was built as part of hands-on Machine Learning practice focusing on real-world problem solving and system design.

---

## 📬 Contact

If you like this project, feel free to connect and collaborate!
