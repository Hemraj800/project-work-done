#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv(r'C:\Users\HP\Downloads\grades.csv')
df


# In[4]:


df.shape


# In[5]:


df.shape[0]  # total rows


# In[6]:


df.shape[1]  # total columns


# In[7]:


df.isnull()


# In[8]:


df.isnull().sum()


# In[9]:


plt.figure(figsize=(15,8))
sns.heatmap(df.isnull())


# In[10]:


df.dtypes


# In[11]:


df['CS-406'].value_counts()


# In[12]:


df['CS-406']=df['CS-406'].fillna('A-')


# In[13]:


df['CS-301'].value_counts()


# In[14]:


df['CS-301']=df['CS-301'].fillna('A-')


# In[15]:


df['HS-304'].value_counts()


# In[16]:


df['HS-304']=df['HS-304'].fillna('A-')


# In[17]:


df['EF-303'].value_counts()


# In[18]:


df['EF-303']=df['EF-303'].fillna('B')


# In[19]:


df['CS-302'].value_counts()


# In[20]:


df['CS-302']=df['CS-302'].fillna('A-')


# In[21]:


df['TC-383'].value_counts()


# In[22]:


df['TC-383']=df['TC-383'].fillna('A')


# In[23]:


df['TC-383'].mode()


# In[24]:


df['MT-442'].value_counts()


# In[25]:


df['MT-442'].mode()


# In[26]:


df['MT-442']=df['MT-442'].fillna('A-')
df['EL-332'].value_counts()


# In[27]:


df['EL-332']=df['EL-332'].fillna('A-')
df['CS-318'].value_counts()


# In[28]:


df['CS-318']=df['CS-318'].fillna('A-')
df['CS-306'].value_counts()


# In[29]:


df['CS-306']=df['CS-306'].fillna('A-')
df['CS-312'].value_counts()


# In[30]:


df['CS-312']=df['CS-312'].fillna('A+')
df['CS-317'].value_counts()


# In[31]:


df['CS-317']=df['CS-317'].fillna('B-')
df['CS-403'].value_counts()


# In[32]:


df['CS-403']=df['CS-403'].fillna('A')
df['CS-421'].value_counts()


# In[33]:


df['CS-421']=df['CS-421'].fillna('B')
df['CS-414'].value_counts()


# In[34]:


df['CS-414']=df['CS-414'].fillna('A')
df['CS-419'].value_counts()


# In[35]:


df['CS-419']=df['CS-419'].fillna('A-')
df['CS-423'].value_counts()


# In[36]:


df['CS-423']=df['CS-423'].fillna('A-')
df['CS-412'].value_counts()


# In[37]:


df['CS-412']=df['CS-412'].fillna('A-')
df.dropna(inplace=True)
df.isnull().sum()


# In[38]:


df.shape


# In[39]:


df.columns


# In[40]:


df.info()


# In[41]:


df.dtypes


# In[42]:


df.drop(['Seat No.'],axis=1,inplace=True)
df.head()


# In[43]:


df.columns


# In[44]:


uniques=['PH-121', 'HS-101', 'CY-105', 'HS-105/12', 'MT-111', 'CS-105', 'CS-106',
       'EL-102', 'EE-119', 'ME-107', 'CS-107', 'HS-205/20', 'MT-222', 'EE-222',
       'MT-224', 'CS-210', 'CS-211', 'CS-203', 'CS-214', 'EE-217', 'CS-212',
       'CS-215', 'MT-331', 'EF-303', 'HS-304', 'CS-301', 'CS-302', 'TC-383',
       'MT-442', 'EL-332', 'CS-318', 'CS-306', 'CS-312', 'CS-317', 'CS-403',
       'CS-421', 'CS-406', 'CS-414', 'CS-419', 'CS-423', 'CS-412']

for i in uniques:
    print(str(i),'=',df[i].unique(),'in this column has total',len(df[i].unique()),'unique values.','\n')


# In[45]:


Count=['PH-121', 'HS-101', 'CY-105', 'HS-105/12', 'MT-111', 'CS-105', 'CS-106',
       'EL-102', 'EE-119', 'ME-107', 'CS-107', 'HS-205/20', 'MT-222', 'EE-222',
       'MT-224', 'CS-210', 'CS-211', 'CS-203', 'CS-214', 'EE-217', 'CS-212',
       'CS-215', 'MT-331', 'EF-303', 'HS-304', 'CS-301', 'CS-302', 'TC-383',
       'MT-442', 'EL-332', 'CS-318', 'CS-306', 'CS-312', 'CS-317', 'CS-403',
       'CS-421', 'CS-406', 'CS-414', 'CS-419', 'CS-423', 'CS-412']

for i in Count:
    print(str(i))
    sns.countplot(df[i])
    plt.show()


# In[46]:


sns.histplot(df['CGPA'])


# In[47]:


from sklearn.preprocessing import LabelEncoder # import


# In[48]:


le=LabelEncoder()
for i in df.drop(['CGPA'],axis=1):
    df[i]=le.fit_transform(df[i])
df


# In[49]:


df.dtypes


# In[50]:


# plot graph for co-relation in Bi Variate Analysis
for col in df.drop(['CGPA'],axis=1):
    plt.figure(figsize=(6,4))
    plt.title('CGPA')
    sns.scatterplot(df['CGPA'],df[col],hue=df['CGPA'])
    plt.show()


# In[51]:


# plot graph for co-relation in Bi Variate Analysis
for col in df.drop(['CGPA'],axis=1):
    plt.figure(figsize=(6,4))
    sns.catplot(x=col,y='CGPA',data=df,kind='bar')
    plt.show()


# In[52]:


df.corr()


# In[53]:


plt.figure(figsize=(50,35))
sns.heatmap(df.corr(),annot=True)


# In[54]:


df.drop(['CGPA'],axis=1).corr()


# In[55]:


plt.figure(figsize=(50,35))
sns.heatmap(df.drop(['CGPA'],axis=1).corr(),annot=True)


# In[56]:


df.describe()


# In[57]:


x=df.drop(['CGPA'],axis=1)
y=df['CGPA']


# In[58]:


#for scaling we use standard scaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
x


# In[59]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split


# In[60]:


for i in range(0,1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=i,test_size=0.25)
    lr.fit(x_train,y_train)
    pred_train=lr.predict(x_train)
    pred_test=lr.predict(x_test)
    if round(r2_score(y_train,pred_train)*100,1)==round(r2_score(y_test,pred_test)*100,1):
        print('At random state',i,'The model perform very well')
        print('Random State = ',i)
        print("Training r2_score is = ",r2_score(y_train,pred_train))
        print("Test r2_score is = ",r2_score(y_test,pred_test))
        print('\n')


# In[ ]:




