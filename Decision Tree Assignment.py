#!/usr/bin/env python
# coding: utf-8

# In[417]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris


# In[ ]:





# In[418]:


iris = load_iris()
type(iris)

# import os
# os.getcwd()


# In[419]:


print (iris.data)
iris.data.shape


# In[420]:


# file = pd.read_csv(r"C:\Users\HP\Downloads\iris.csv")
# print(file.columns.values)


# In[421]:


print (iris.feature_names)


# In[422]:


print (iris.target)


# In[423]:


print (iris.target_names)


# In[424]:


type('iris.data')
type('iris.target')


# In[425]:


iris.data.shape


# In[426]:


iris.target.shape


# In[427]:


featuresAll=[]
features = iris.data[: , [0,1,2,3]]
features.shape


# In[428]:


targets = iris.target
targets.reshape(targets.shape[0],-1)
targets.shape


# In[429]:


for observation in features:
    featuresAll.append([observation[0] + observation[1] + observation[2] + observation[3]])
print (featuresAll)


# In[ ]:





# In[ ]:





# In[ ]:





# In[430]:


# X = ['sepal.length','sepal.width','petal.length','petal.width']
# Y = ['sepal.length','sepal.width','petal.length','petal.width']


# In[431]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3)


# In[432]:


train.shape,test.shape


# In[ ]:





# In[487]:


train_X=[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
# train_X=train[('Petal length' ,'Petal Width' ,'Sepal Length' ,'Sepal Width')]
train_y=train.variety
test_X=test[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
test_y=test.variety


# In[ ]:





# In[441]:


# dtmodel=DecisionTreeClassifier()
# dtmodel.fit(train_X,train_y)

# dtpredict=dtmodel.predict(test_X)

# dtaccuracy=metrics.accuracy_score(dtpredict,test_y)
# print('Decission Tree Model Accuracy is {}'.format(dtaccuracy * 100))

# test_preddf=test.copy()
# test_preddf['predicted variety']=dtpredict
# wrongpred=test_preddf.loc[test['variety'] != dtpredict]
# wrongpred


# In[434]:


dataset[4].unique()


# In[ ]:





# In[ ]:


def importdata():
    iris_data = pd.read_csv(r'C:\Users\HP\Downloads\iris.csv', sep=',', header = None)
    
    # Printing the dataset shape
    print("Dataset Length: ",len(iris_data))
    print("Dataset Shape: ", iris_data.shape)
    
    # Printing dataset observation
    print("Dataset: ", iris_data.head())
    return iris_data


# In[ ]:


def splitdataset(iris_data):
    
    # Seperating the target variable
    X = iris_data.values[:, 1:5]
    Y = iris_data.values[:, 0]
    
    # Spliting the datset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
    
    return X, Y, X_train, X_test, y_train, y_test


# In[456]:


dataset = importdata()
dataset.head()


# In[479]:


def train_using_gini(X_train, X_test, y_train):
    
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
    
    # Performing Training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# In[467]:


from sklearn import preprocessing
def train_using_entropy(X_train, X_test, y_train):
    
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
    

def convert(iris):
    number = preprocessing.LabelEncoder()
    iris['variety'] = number.fit_transform(iris['variety'])
    iris=iris.fillna(-999) # fill holes with default value
    return iris
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# In[ ]:





# In[ ]:





# In[473]:


def prediction(X_test, clf_object):
    
    # Prediction on test using giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values: ")
    print(y_pred)
    return y_pred


# In[474]:


def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test,y_pred))
    print("Accuracy: ",accuracy_score(y_test,y_pred)*100)
    print("Report: ",classification_report(y_test, y_pred))
    


# In[481]:


def main():
    
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test,y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train,X_test,y_train)
    
    #Operational Phase
    print("Results Using Gini Index: ")
    
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    print("Results Using Entropy: ")
    
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


# In[477]:





# In[483]:


# if __name__ =="__main__":
#     main()


# In[ ]:





# In[ ]:




