#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df_tr=pd.read_csv(r'C:\Users\HP\Downloads\termdeposit_train.csv')
df_tr


# In[3]:


df_tr.shape


# In[4]:


df_tr.isnull()


# In[5]:


df_tr.isnull().sum()


# In[6]:


sns.heatmap(df_tr.isnull())
plt.show()


# In[7]:


df_tr.isna().sum()


# In[8]:


df_tr.columns


# In[9]:


df_tr.info()


# In[10]:


df_tr.dtypes


# In[11]:


df_tr.drop(['ID'],axis=1,inplace=True)
df_tr.head()


# In[12]:


df_tr.columns


# In[13]:


df_tr['age'].unique()


# In[14]:


df_tr['job'].unique()


# In[15]:


df_tr['marital'].unique()


# In[16]:


df_tr['education'].unique()


# In[17]:


df_tr['default'].unique()


# In[18]:


df_tr['balance'].unique()


# In[19]:


df_tr['housing'].unique()


# In[20]:


df_tr['loan'].unique()


# In[21]:


df_tr['contact'].unique()


# In[22]:


df_tr['day'].unique()


# In[23]:


df_tr['month'].unique()


# In[24]:


df_tr['duration'].unique()


# In[25]:


df_tr['campaign'].unique()


# In[26]:


df_tr['pdays'].unique()


# In[27]:


df_tr['previous'].unique()


# In[28]:


df_tr['poutcome'].unique()


# In[29]:


df_tr['subscribed'].unique()


# In[30]:


sns.histplot(df_tr['age'])
plt.show()


# In[31]:


df_tr.columns


# In[32]:


print(df_tr['job'].value_counts(),'\n')
print(df_tr['job'].value_counts(normalize=True)*100)


# In[33]:


sns.histplot(df_tr['marital'])
print(df_tr['marital'].value_counts())
print(df_tr['marital'].value_counts(normalize=True)*100)


# In[34]:


sns.histplot(df_tr['education'])
print(df_tr['education'].value_counts())
print(df_tr['education'].value_counts(normalize=True)*100)


# In[35]:


sns.histplot(df_tr['default'])
print(df_tr['default'].value_counts())
print(df_tr['default'].value_counts(normalize=True)*100)


# In[36]:


plt.plot(figsize=(7,9))
plt.hist(df_tr['balance'])
plt.show()


# In[37]:


sns.histplot(df_tr['housing'])
print(df_tr['housing'].value_counts())
print(df_tr['housing'].value_counts(normalize=True)*100)


# In[38]:


sns.histplot(df_tr['loan'])
print(df_tr['loan'].value_counts())
print(df_tr['loan'].value_counts(normalize=True)*100)


# In[39]:


sns.histplot(df_tr['contact'])
print(df_tr['contact'].value_counts())
print(df_tr['contact'].value_counts(normalize=True)*100)


# In[40]:


sns.histplot(df_tr['day'])


# In[41]:


sns.histplot(df_tr['month'])
print(df_tr['month'].value_counts())
print(df_tr['month'].value_counts(normalize=True)*100)


# In[42]:


plt.hist(df_tr['duration'])
plt.show()


# In[43]:


sns.countplot(df_tr['campaign'])
print(df_tr['campaign'].value_counts())
print(df_tr['campaign'].value_counts(normalize=True)*100)


# In[44]:


sns.histplot(df_tr['pdays'])


# In[45]:


sns.histplot(df_tr['poutcome'])
print(df_tr['poutcome'].value_counts())
print(df_tr['poutcome'].value_counts(normalize=True)*100)


# In[46]:


sns.histplot(df_tr['subscribed'])
print(df_tr['subscribed'].value_counts())
print(df_tr['subscribed'].value_counts(normalize=True)*100)


# In[47]:


from sklearn.preprocessing import LabelEncoder # import


# In[48]:


df_tr.dtypes


# In[49]:


le=LabelEncoder()
for i in df_tr.drop(['subscribed'],axis=1):
    df_tr[i]=le.fit_transform(df_tr[i])
df_tr


# In[50]:


df_tr.head()


# In[51]:


df_tr.dtypes


# In[53]:


# plot graph for co-relation in Bi Variate Analysis
import seaborn as sns
for col in df_tr.drop(['subscribed'],axis=1):
    plt.figure(figsize=(6,4))
    plt.title(f'{col} vs. subscribed')
    sns.scatterplot(y=df_tr[col],x=df_tr['subscribed'],hue=df_tr['subscribed'])
    plt.show()


# In[54]:


plt.figure(figsize=(6,4))
sns.catplot(x='job',y='subscribed',data=df_tr,kind='bar')
plt.show()


# In[55]:


plt.figure(figsize=(6,4))
sns.catplot(x='day',y='subscribed',data=df_tr,kind='bar')
plt.show()


# In[56]:


plt.figure(figsize=(6,4))
sns.catplot(x='month',y='subscribed',data=df_tr,kind='bar')
plt.show()


# In[57]:


df_tr.corr()


# In[58]:


plt.figure(figsize=(15,8))
sns.heatmap(df_tr.corr(),annot=True)


# In[ ]:




