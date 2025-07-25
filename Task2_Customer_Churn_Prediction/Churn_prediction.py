#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("Churn_modelling.csv")
data


# In[2]:


data.isnull().sum()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


# data preprocessing
print("Number of unique values: ", len(data["Geography"].unique()),'\nList of unique values: ',data["Geography"].unique())


# In[6]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Geography"]=le.fit_transform(data["Geography"])
print("Number of unique values : ",len(data["Geography"].unique()),'\nList of unique values:',data["Geography"].unique())


# In[7]:


print("Number of unique values: ", len(data["Gender"].unique()),'\nList of unique values: ',data["Gender"].unique())


# In[8]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Gender"]=le.fit_transform(data["Gender"])
print("Number of unique values : ",len(data["Gender"].unique()),'\nList of unique values:',data["Gender"].unique())


# In[9]:


print("Number of unique values: ", len(data["Surname"].unique()),'\nList of unique values: ',data["Surname"].unique())


# In[10]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
data.iloc[:,2]=np.array(ct.fit_transform(data))
data["Surname"]


# In[11]:


data.info()


# In[12]:


data


# In[13]:


print(data.dtypes)


# In[20]:


# Feature Extraction 
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.pairplot(data)
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import issparse

# Step 1: Identify and remove sparse matrix columns
for col in data.columns:
    if data[col].apply(lambda x: issparse(x)).any():
        print(f"Column '{col}' contains sparse data. Dropping it.")
        data = data.drop(columns=[col])

# Step 2: Keep only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Step 3: Plot the numeric data
sns.pairplot(numeric_data)
plt.show()


# In[21]:


import warnings 
warnings.filterwarnings("ignore")
plt.figure(figsize=[15,8])
sns.heatmap(data.corr(),annot=True)
plt.show()


# In[22]:


data.columns


# In[24]:


J=data.drop(columns=['Gender']).iloc[:,2:-1]
A=data.iloc[:,-1].values


# In[25]:


J


# In[26]:


A


# In[27]:


import warnings
warnings.filterwarnings("ignore")
plt.figure(figsize=[15,8])
sns.heatmap(data.corr(),annot=True)
plt.show


# In[30]:


from sklearn.model_selection import train_test_split
j_train,j_test,a_train,a_test=train_test_split(J,A,test_size=0.3,random_state=42)
print("Shape of j_train : ",j_train.shape)
print("Shape of j_test : ",j_test.shape)
print("Shape of a_train : ",a_train.shape)
print("Shape of a_test : ",a_test.shape)


# In[33]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(class_weight='balanced')
lr.fit(j_train,a_train)


# In[34]:


# Model prediction and evaluation
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score,f1_score,confusion_matrix
import numpy as np
test_score=lr.score(j_test,a_test)
train_score=lr.score(j_train,a_train)
print("Testing score: ",test_score)
print("Train score: ",train_score)


# In[35]:


a_pre=lr.predict(j_test)
a_pre


# In[37]:


r2s=r2_score(a_test,a_pre)
mae=mean_absolute_error(a_test,a_pre)
mse=mean_absolute_error(a_test,a_pre)
rmse=np.sqrt(mse)
accsc=accuracy_score(a_test,a_pre)
f1s=f1_score(a_test,a_pre)
cm=confusion_matrix(a_test,a_pre)
print("R2 score : ",r2s)
print("Mean absolute error : ",mae)
print("Mean Squared error : ",mse)
print("Root Mean Squared error :",rmse)
print("Accuracy score : ",f1s)
print("Confusion Matrix : \n",cm)


# In[38]:


# Model building
from sklearn.model_selection import train_test_split
j_train,j_test,a_train,a_test=train_test_split(J,A, test_size=0.3,random_state=42)
print("Shape of j_train:",j_train.shape)
print("Shape of j_test:",j_test.shape)
print("Shape of a_train:",a_train.shape)
print("Shape of a_train:",a_test.shape)


# In[39]:


from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(class_weight='balanced')
RFC.fit(j_train,a_train)


# In[40]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score,f1_score,confusion_matrix
import numpy as np
test_score=RFC.score(j_test,a_test)
train_score=RFC.score(j_train,a_train)
print("Testing score: ",test_score)
print("Train score: ",train_score)


# In[41]:


a_pre=RFC.predict(j_test)
a_pre


# In[42]:


a_test


# In[51]:


r2s=r2_score(a_test,a_pre)
mae=mean_absolute_error(a_test,a_pre)
mse=mean_absolute_error(a_test,a_pre)
rmse=np.sqrt(mse)
accsc=accuracy_score(a_test,a_pre)
f1s=f1_score(a_test,a_pre)
cm=confusion_matrix(a_test,a_pre)
print("R2 score : ",r2s)
print("Mean absolute error : ",mae)
print("Mean Squared error : ",mse)
print("Root Mean Squared error :",rmse)
print("Accuracy score : ",accsc)
print("F1 score: " ,f1s)
print("Confusion Matrix : \n",cm)


# In[44]:


# Model Building 
from sklearn.model_selection import train_test_split
j_train,j_test,a_train,a_test=train_test_split(J,A, test_size=0.3,random_state=42)
print("Shape of j_train:",j_train.shape)
print("Shape of j_test:",j_test.shape)
print("Shape of a_train:",a_train.shape)
print("Shape of a_train:",a_test.shape)


# In[46]:


from sklearn.ensemble import GradientBoostingClassifier
GBC=GradientBoostingClassifier()
GBC.fit(j_train,a_train)


# In[47]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score,f1_score,confusion_matrix
import numpy as np
test_score=GBC.score(j_test,a_test)
train_score=GBC.score(j_train,a_train)
print("Testing score: ",test_score)
print("Train score: ",train_score)


# In[48]:


a_pre=GBC.predict(j_test)
a_pre


# In[50]:


r2s=r2_score(a_test,a_pre)
mae=mean_absolute_error(a_test,a_pre)
mse=mean_absolute_error(a_test,a_pre)
rmse=np.sqrt(mse)
accsc=accuracy_score(a_test,a_pre)
f1s=f1_score(a_test,a_pre)
cm=confusion_matrix(a_test,a_pre)
print("R2 score : ",r2s)
print("Mean absolute error : ",mae)
print("Mean Squared error : ",mse)
print("Root Mean Squared error :",rmse)
print("Accuracy score : ",accsc)
print("F1 score: " ,f1s)
print("Confusion Matrix : \n",cm)


# In[52]:


# Model testing
j_test.iloc[0]


# In[ ]:




