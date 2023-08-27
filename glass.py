#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


# In[32]:


df=pd.read_csv(r'C:\Users\HP\Downloads\glass (3).csv')
df


# In[4]:


df.shape


# In[5]:


df.shape[0]


# In[6]:


df.shape[1]


# In[7]:


df.isnull()


# In[8]:


df.isnull().sum()


# In[9]:


df.isna().sum()


# In[10]:


sns.heatmap(df.isnull())


# In[11]:


df.columns


# In[12]:


df.info()


# In[13]:


df.dtypes


# In[15]:


df.head()


# In[17]:


sns.heatmap(df.isnull())


# In[18]:


df.columns


# In[19]:


df.info()


# In[20]:


df.dtypes


# In[22]:


df.head()


# In[27]:


df.columns


# In[ ]:




