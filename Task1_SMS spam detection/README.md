# Task 1: SMS Spam Detection

## ✅ Objective
To build a machine learning model that classifies SMS messages as either **spam** or **ham (not spam)** using natural language processing techniques.

## 🧠 Dataset
Dataset used: SMS Spam Collection Dataset  
Link: [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## 🔧 Libraries Used
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  
- nltk (Natural Language Toolkit)  
- wordcloud

## 🪜 Steps Followed
1. Loaded and explored the SMS spam dataset.
2. Cleaned and preprocessed the text (lowercasing, removing stopwords, punctuation).
3. Converted text into numerical features using TF-IDF vectorization.
4. Split the dataset into training and test sets.
5. Trained multiple classifiers (e.g., Naive Bayes, SVM, Logistic Regression).
6. Evaluated performance using accuracy, precision, recall, and F1-score.
7. Visualized confusion matrix and most frequent spam words.

## 🧾 Sample Code
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label_num']  # 0 for ham, 1 for spam

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds))
