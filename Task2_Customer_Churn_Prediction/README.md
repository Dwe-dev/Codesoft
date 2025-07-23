# Task 2: Customer Churn Prediction

## ✅ Objective
Build a machine learning model to predict whether a customer will leave a service (churn) based on past usage and profile.

## 🧠 Dataset
Dataset used: Telco Customer Churn Dataset  
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  

## 🔧 Libraries Used
- pandas, numpy, seaborn, matplotlib
- scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier)

## 🪜 Steps Followed
1. Loaded and cleaned the customer data.
2. Performed EDA to understand churn behavior.
3. Converted categorical columns using encoding.
4. Split the dataset into training/testing sets.
5. Trained multiple models and compared results.
6. Evaluated with metrics: accuracy, confusion matrix, precision, recall.

## 🧾 Sample Code
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv('churn.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

