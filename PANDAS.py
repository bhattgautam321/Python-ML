#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
pandas.__version__
import os
os.getcwd()


# In[117]:


Data = pd.read_csv(r'C:\Users\HP\Downloads\Csvfiles\carsales.csv')
Data.head(4)
Data.tail(4)
Data.dtypes


# In[118]:


Data.shape


# In[120]:


#Copied file
x = Data.copy()
x.head()


# In[121]:


x.columns


# In[122]:


x.columns = ['a','b','c','d','e']
x.columns


# In[123]:


x.info()


# In[125]:


#analysis of numerical data

x.describe()


# In[126]:


x.drop('c',axis=1, inplace = True)
x.head()


# In[127]:


#For accessing a particular column

x.b


# In[128]:


x['b']


# In[129]:


#For accessing the data rows

print(x.loc[2:5])


# In[130]:


#To access a desired range of data

x.iloc[2:4,2:6]


# In[131]:


print(x.isnull().sum())


# In[132]:


import numpy as np
x.iloc[2:6,4:5]
print(x.head(10))


# In[135]:


#This command will put the mean value but here the values are in %age so it will show type error


x.e.fillna(x.e.mean(),inplace = True)
print(x.head(6))


# In[136]:


print(x[x.e>1])
print(x[x.e>1].describe())


# In[141]:


def change(text):
    if text == 'Fiet':
        return 69520
    elif text == 'Ford':
        return 615184
    elif text == 'Mercedes':
        return 135183
    elif text == 'Vauxhall':
        return 22121248
    else:
        return 14919
    
x['a'] = x.a.apply(change)
x.head(6)


# In[143]:


#ADD ROWS AND COLUMN AND SAMPLE POINT

x.iloc[0] = ['Fiet',1800,1100,100]
x.head(5)


# In[154]:


import random
x['New Column'] = random.randrange(9)
x.head()


# In[172]:


#FOR INSERTING A COLUMN AT GIVEN POSITION

#x.insert(2, 'Vers Sports as 1 else 0','Default')
x.head()


# In[173]:


s = pd.Series(list('abacd'))
s


# In[174]:


dummy_data = pd.get_dummies(s)
print(dummy_data)


# In[ ]:




