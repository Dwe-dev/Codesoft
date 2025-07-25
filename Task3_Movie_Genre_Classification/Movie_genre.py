#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
os.listdir()


# In[2]:


import pandas as pd
import warnings
data=pd.read_csv("description.txt")
data


# In[3]:


# create and apply function to read data by splitting
def load_data(file_path):
    with open(file_path,'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [line.strip().split(' ::: ') for line in data]
    return data


# In[4]:


train_data = load_data("train_data.txt")
train_df= pd.DataFrame(train_data, columns=['ID','Title','Genre','Description'])
test_data=load_data("test_data.txt")
test_df= pd.DataFrame(test_data,columns=['ID','Title','Description'])
test_solution=load_data('test_data_solution.txt')
test_solution_df=pd.DataFrame(test_solution,columns=['ID','Title','Genre','Description'])


# In[5]:


print("Train Data:")
train_df 


# In[6]:


print("\nTest Data:")
test_df


# In[7]:


print("\nTest solution:")
test_solution_df


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 10000)
J_train_tfidf=vectorizer.fit_transform(train_df["Description"])
J_test_tfidf= vectorizer.transform(test_df["Description"])
print(f"Training data shape: {J_train_tfidf.shape}")
print(f"Test data shape: {J_test_tfidf.shape}")


# In[9]:


# Encoding the target names
from sklearn.preprocessing import LabelEncoder
names_encoder= LabelEncoder()
A_train = names_encoder.fit_transform(train_df['Genre'])
print(f"Unique genres in the training data : {names_encoder.classes_}")


# In[10]:


# Model training 
from sklearn.linear_model import LogisticRegression
lo_model= LogisticRegression(max_iter = 300)
lo_model.fit(J_train_tfidf,A_train)
A_pre= lo_model.predict(J_test_tfidf)
predicted_genres = names_encoder.inverse_transform(A_pre)
test_df['Predicted_Genre']=predicted_genres
test_df[['Title','Predicted_Genre']]


# In[11]:


test_df['Predicted_Genre'] = predicted_genres
merged_df=pd.merge(test_solution_df[['ID','Genre']],test_df[['ID','Predicted_Genre']],on='ID')
merged_df


# In[12]:


# Model Evaluation
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(merged_df['Genre'],merged_df['Predicted_Genre'])
print(f"Accuracy : {accuracy: .4f}")
print("\nClassification Report:")
print(classification_report(merged_df['Genre'],merged_df['Predicted_Genre'], zero_division=0))


# In[13]:


from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(J_train_tfidf,A_train)


# In[14]:


a_pre_nb=nb_model.predict(J_test_tfidf)
predicted_genres_nb = names_encoder.inverse_transform(a_pre_nb)
test_df['Predicted_Genre_NB'] = predicted_genres_nb
merged_df_nb = pd.merge(test_solution_df, test_df[['ID', 'Predicted_Genre_NB']],on='ID')


# In[15]:


from sklearn.metrics import accuracy_score, classification_report
accuracy_nb = accuracy_score(merged_df_nb['Genre'],merged_df_nb['Predicted_Genre_NB'])
print(f"Naive Bayes Accuracy : {accuracy_nb}")
print("Naive Bayes Classification Report:")
print(classification_report(merged_df_nb['Genre'],merged_df_nb['Predicted_Genre_NB'], target_names=names_encoder.classes_,zero_division=0))


# In[16]:


# Model building: SVM
from sklearn.svm import LinearSVC
svm_model = LinearSVC(max_iter=2000)
svm_model.fit(J_train_tfidf,A_train)


# In[17]:


a_pre_svm=svm_model.predict(J_test_tfidf)
predicted_genres_svm = names_encoder.inverse_transform(a_pre_svm)
test_df['Predicted_Genre_SVM'] = predicted_genres_svm
merged_df_svm = pd.merge(test_solution_df, test_df[['ID', 'Predicted_Genre_SVM']],on='ID')


# In[18]:


from sklearn.metrics import accuracy_score, classification_report
accuracy_svm = accuracy_score(merged_df_svm['Genre'],merged_df_svm['Predicted_Genre_SVM'])
print(f"SVM Accuracy : {accuracy_svm}")
print("SVM Classification Report:")
print(classification_report(merged_df_svm['Genre'],merged_df_svm['Predicted_Genre_SVM'], target_names=names_encoder.classes_,zero_division=0))


# In[ ]:




