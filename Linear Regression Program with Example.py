#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Single Variable

import numpy as np
import matplotlib.pyplot as plt
def estimate_coef(x,y):
    

#Number of observations/points
    n = np.size(x)

# Mean of x and y vector    
    m_x,m_y = np.mean(x),np.mean(y)
    
# Calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x-n*m_y*m_x)
    SS_xx = np.sum(x*x-n*m_x*m_x)
    
# Calculating regression Coefficient
    b_1 = SS_xy/SS_xx
    b_0 = m_y-b_1*m_x
    return(b_0,b_1)

def plot_regression_line(x,y,b):
    
#Plotting the actual points as scatter plot
    plt.scatter(x,y,color='m',marker='o',s=30)
    
# Predicted response vector
    y_pred = b[0]+b[1]*x
    
# Plotting the regression line
    plt.plot(x,y_pred, color="g")
    
# Putting Labels
    plt.xlabel('x')
    plt.ylabel('y')
    
# Function to show plot
    plt.show()
    
def main():
    
# Observations
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([1,3,2,5,7,8,8,9,10,12])
    
# Estimating Coefficients
    b = estimate_coef(x,y)
    print("Estimated coeffiecients:\n b_0 = {} \\n b_1 = {}".format(b[0],b[1]))
    
# Plotting Regression Line

    plot_regression_line(x,y,b)
if __name__ == "__main__":
    main()


# In[21]:


# Multiple Variable

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,metrics

#load the boston dataset
boston = datasets.load_boston(return_X_y = False)

# Defining feature matrix(X) and response vector(y)
X = boston.data
y = boston.target

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





# In[ ]:




