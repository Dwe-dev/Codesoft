#!/usr/bin/env python
# coding: utf-8

# In[49]:


# Import modules 
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords


# In[28]:


# Loading the dataset
df=pd.read_csv('spam.csv',encoding='latin-1')
df.head()


# In[29]:


import os
print(os.getcwd())


# In[30]:


import nltk
nltk.download('stopwords')


# In[39]:


# necessary columns
df=df[['v1','v2']]
# df.rename(coulmns={'v2' : 'sms', 'v1': ' names'}, inplace=True)
df = df.rename(columns={'v2':'messages' , 'v1' : 'names'})
df.head()


# In[40]:


# preprocessing the dataset(check for null values)
df.isnull().sum()


# In[46]:


STOPWORDS = set(stopwords.words('english'))
import re
def cln_txt(text):
    text = text.lower()
    text= re.sub(r'[^0-9a-zA-Z]', ' ', text)
    text=re.sub(r'\s+', ' ',text)
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text


# In[47]:


# cln the messages
df['cln_txt'] = df['messages'].apply(cln_txt)
df.head()


# In[48]:


# input split
J= df['cln_txt']
A=df['names']


# In[55]:


# Model Training
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

def classi_fy(model,J,A):
    # train test split
    j_train,j_test,a_train,a_test=train_test_split(J,A,test_size=0.25, random_state=42, shuffle=True, stratify= A)
    # model training
    pipeline_model = Pipeline([('vect', CountVectorizer()),
                              ('tfidf',TfidfTransformer()),
                              ('clf', model)])
    pipeline_model.fit(j_train,a_train)
    print("Precision:", pipeline_model.score(j_test, a_test)*100)
    a_pre = pipeline_model.predict(j_test)
    print(classification_report(a_test, a_pre))


# In[56]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classi_fy(model,J,A)


# In[57]:


from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB()
classi_fy(model,J,A)


# In[61]:


from sklearn.svm import SVC
model = SVC(C=7)
classi_fy(model,J,A)


# In[ ]:




