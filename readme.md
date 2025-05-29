# 💬 phishGuard - Spam Classifier using Scikit-learn & spaCy

A simple machine learning project to classify SMS messages as **Spam** or **Ham** (Not Spam) using `TF-IDF`, `Logistic Regression`, and basic NLP preprocessing with `spaCy`.  
Built with a minimal interactive frontend using **Streamlit**.

---

## 📚 Dataset Info
Source: Kaggle - [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

5,000+ labeled SMS messages

Label: ham = not spam and spam

---

## 🔍 Features

- TF-IDF vectorization of cleaned messages  
- Logistic Regression model trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- Realtime message classification in a web app  
- Preprocessing pipeline using `spaCy`  
- Model + vectorizer saved via `joblib` for reuse  

---

## 📦 Tech Stack

| Layer         | Tools Used                   |
|---------------|------------------------------|
| Language      | Python 3                     |
| ML Libraries  | Scikit-learn, Pandas, spaCy  |
| Frontend      | Streamlit                    |
---

## 🧠 How It Works

1. Data loaded from the CSV
2. Preprocessing using `spaCy` to clean messages
3. TF-IDF vectoriser transforms text to features
4. Logistic Regression trains on vectorised data
5. Model and vectoriser saved using `joblib`
6. Streamlit frontend loads model, lets user input a message
7. Input is preprocessed -> vectorised -> predicted

---

## 🔧 How to Run Locally

git clone https://github.com/mithurssan/phishGuard.git

cd phishGuard

python -m venv venv
source venv/scripts/activate - on bash

pip install -r requirements.txt

### Train model (if you want to train yourself)
python backend/train.py

### Run the app
streamlit run app.py
