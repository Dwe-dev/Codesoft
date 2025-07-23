# Task 1: Handwritten Text Generation

## ✅ Objective
To build a character-level Recurrent Neural Network (RNN) using LSTM that generates handwritten-like text sequences.

## 🧠 Dataset
Dataset used: IAM Handwriting Database  
Link: https://www.fki.inf.unibe.ch/databases/iam-handwriting-database  

## 🔧 Libraries Used
- pandas, numpy
- matplotlib, seaborn
- tensorflow / keras
- sklearn.metrics

## 🪜 Steps Followed
1. Loaded and preprocessed the handwritten text dataset.
2. Converted characters into integer sequences.
3. Used one-hot encoding for character-level tokenization.
4. Built and trained a LSTM-based RNN model.
5. Used temperature sampling to generate new text.
6. Saved generated samples to a `.txt` file.
7. Plotted training loss curves.

## 🧾 Sample Code
```python
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, vocab_size)))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=20, batch_size=128)

