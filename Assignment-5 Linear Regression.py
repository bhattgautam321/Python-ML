#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,metrics
import os
import pandas as pd

#load the boston dataset
# Data = datasets.load_Data(return_X_y = False)
data = pd.read_csv(r"C:\Users\HP\Downloads\Samplemarks.csv")
# Defining feature matrix(X) and response vector(y)
X = data[['English ', 'Maths','Science']]
y = data['Grades']


#Splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.4,random_state = 1)

# Create Linear regression object
reg = linear_model.LinearRegression()

# Train the model using training sets
reg.fit(X_train, y_train)

# Regression coeffcients
print("Coefficients: \n",reg.coef_)

# Variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# Plot for residual error  ## Setting plot style
plt.style.use("fivethirtyeight")

# Plotting residual errors in training data
plt.scatter(reg.predict(X_train),reg.predict(X_train)-y_train,color = "green", s = 10, label = "Train data")

# Plotting residual errors in test data
plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,color = "blue", s = 10, label = "Test data")

# Plotting line for 0 residual error
plt.hlines(y = 0,xmin = 0,xmax = 50,linewidth = 2)

# Plotting legend
plt.legend(loc = "upper right")

# Plot title 
plt.title("Residual errors")

# Function to show plot
plt.show()


# In[ ]:




