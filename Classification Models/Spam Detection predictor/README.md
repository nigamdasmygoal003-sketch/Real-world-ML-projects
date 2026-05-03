# 💬 Spam Message Detection System

An end-to-end **Machine Learning + NLP project** that classifies messages as **Spam or Ham (Not Spam)** using text processing and probabilistic modeling.

This project demonstrates how to build a **real-world NLP pipeline**, train a model, and deploy it as an interactive **chat-style desktop application**.

---

## 🚀 Features

* 🧠 Text classification using **Naive Bayes**
* 🔤 Feature extraction using **TF-IDF**
* 📊 Uses **unigrams + bigrams** for better context
* ⚡ Fast and lightweight model (BernoulliNB)
* 💾 Model persistence using `joblib`
* 💬 Chat-style UI built with CustomTkinter
* 🔮 Real-time spam prediction with probability

---

## 🧠 Problem Statement

Spam messages are common in communication systems.

Goal:

> Automatically detect whether a message is **spam or legitimate (ham)**.

---

## 📊 Dataset

* SMS Spam Collection Dataset
* Contains labeled messages:

  * `ham` → normal message
  * `spam` → unwanted message

---

## 🏗️ Project Structure

```text
Spam Detection predictor/
│
├── data/
│   └── spam.csv
│
├── model/
│   └── spam_model.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/nigamdasmygoal003-sketch/Real-world-ML-projects
cd "Spam Detection predictor"

pip install -r requirements.txt
```

---

## 🧪 Model Training

```bash
python src/train.py
```

This will:

* Preprocess text data
* Train the model
* Save it to `model/spam_model.pkl`

---

## 🔮 Prediction (CLI)

```bash
python src/predict.py
```

Example output:

```json
{
  "spam": true,
  "spam_probability": 0.91
}
```

---

## 💬 Run Chat Application

```bash
python app.py
```

Features:

* Chat-style interface
* Real-time spam detection
* Probability-based prediction
* User-friendly interaction

---

## 🧠 Model Details

* Algorithm: **Bernoulli Naive Bayes**
* Feature Engineering:

  * TF-IDF Vectorization
  * Stop word removal
  * N-grams (1,2)
* Input: Raw text message
* Output: Spam / Ham + probability

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

## 📌 Key Learnings

* NLP pipeline using TF-IDF
* Text preprocessing and feature extraction
* Model selection (Naive Bayes vs Logistic)
* Building ML pipelines for text data
* Deploying ML model with UI

---

## 🚀 Future Improvements

* Convert to Streamlit web app
* Add REST API (FastAPI)
* Use advanced NLP models (BERT)
* Improve UI/UX design
* Add message history storage

---

## 🧑‍💻 Author

Nigam Das
Machine Learning Enthusiast 🚀

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
