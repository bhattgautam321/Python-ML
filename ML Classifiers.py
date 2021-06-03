#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn


# In[2]:


sklearn.__version__


# In[11]:


# 1 for smmoth and for apple 0 for bumpy and for Orange
from sklearn import tree
features = [[140,"1"],[130,"1"],[150,"0"],[170,"0"]]
labels = [1,1,0,0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print (clf.predict([[160,0]]))


# In[13]:


from sklearn import tree
Horsepower = [[300,2],[450,2],[200,8],[150,9]]
labels = ["sports-car","sports-car","minivan","minivan"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Horsepower,labels)
print(clf.predict([[200,6]]))


# In[ ]:




