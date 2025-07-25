# 📩 SMS Spam Detection - CodeSoft Internship Task 1

This project is part of **Task 1** for the **@CodeSoft** internship. It aims to build a **Spam Detection System** using Machine Learning and Natural Language Processing (NLP) techniques.

---

## 🚀 Project Overview

The goal is to classify SMS messages as **Spam** or **Not Spam (Ham)** using a trained ML model. The solution involves:
- Text cleaning
- Tokenization
- Vectorization (TF-IDF)
- Model training & evaluation

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Matplotlib & Seaborn (for visualization)
- Jupyter Notebook / VS Code

---

## 📂 Dataset

The dataset used is a public SMS Spam Collection dataset with the following structure:

- **Labels**: `ham` (not spam), `spam` (unwanted text)

---

## 🔍 Workflow

1. **Data Preprocessing**
   - Lowercasing
   - Removing punctuation, stopwords
   - Tokenization
   - Stemming (using NLTK)

2. **Feature Extraction**
   - Using **TF-IDF Vectorizer**

3. **Model Training**
   - Used algorithms:
     - Naive Bayes
     - Logistic Regression
     - SVM

4. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Precision, Recall, F1 Score

---

## 📈 Results

- Achieved **~97% accuracy** using the **Multinomial Naive Bayes** classifier.
- High precision and recall for both classes.

---

## 📁 Project Structure
── SMS Spam Detection.py
├── spam.csv
├── README.md
