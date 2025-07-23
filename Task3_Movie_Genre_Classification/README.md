# Task 3: Movie Genre Classification

## ✅ Objective
Build a machine learning model that can classify the genre of a movie based on its plot summary using NLP techniques.

## 🧠 Dataset
Dataset used: **Genre Classification Dataset – IMDb**  
📎 Kaggle link: [https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

The dataset includes:
- Movie titles
- Plot summaries
- Corresponding genre labels

## 🔧 Libraries Used
- pandas
- numpy
- scikit-learn (TfidfVectorizer, LogisticRegression, NaiveBayes, SVM)
- matplotlib, seaborn

## 🪜 Steps Followed
1. Loaded and inspected the IMDb genre dataset.
2. Cleaned the text data (removed punctuation, stopwords, lowercased).
3. Converted text data into numeric features using `TfidfVectorizer`.
4. Split the data into training and test sets.
5. Trained multiple models like:
   - Logistic Regression
   - Naive Bayes
   - Support Vector Machine (SVM)
6. Evaluated each model’s accuracy, F1-score, and confusion matrix.
7. Compared results and chose the best performing model.

## 🧾 Sample Code
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('imdb.csv')
X = df['plot']
y = df['genre']

# Convert plot summaries into TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vect = vectorizer.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

