#!/usr/bin/env python
# coding: utf-8

# In[44]:


import matplotlib
print(matplotlib.__version__)

import matplotlib.pyplot as plt

import numpy as np

plt.plot([-9,9,7,0,56,4])
plt.show()


# In[7]:


plt.plot([-9,9,7,0,56,4],"or")
plt.show()


# In[12]:


days = list(range(0,23,3))
temp = [23,25,30,31,29,35,25,39]
plt.plot(days, temp,"g")
plt.show()


# In[15]:


plt.scatter(days, temp)


# In[16]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]
plt.plot(days,temp_min)
plt.show()

plt.plot(days,temp_max)
plt.show()


# In[17]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]

plt.plot(days, temp_min)
plt.plot(days, temp_max)
plt.show()


# In[19]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]
plt.plot(days, temp_min,"b")
plt.plot(days, temp_min,"ob")
plt.plot(days, temp_max,"r")
plt.plot(days, temp_max,"or")
plt.show()


# In[23]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]
plt.plot(days, temp_min,"b")
plt.plot(days, temp_min,"ob")
plt.plot(days, temp_max,"r")
plt.plot(days, temp_max,"or")
plt.xlabel('days')
plt.ylabel('temperature')
plt.show()


# In[27]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]
plt.plot(days, temp_min,"b")
plt.plot(days, temp_min,"ob")
plt.plot(days, temp_max,"r")
plt.plot(days, temp_max,"or")
plt.xlabel('days')
plt.ylabel('temperature')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)


# In[34]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]
plt.plot(days, temp_min,"b")
plt.plot(days, temp_min,"ob")
plt.plot(days, temp_max,"r")
plt.plot(days, temp_max,"or")
plt.xlabel('days')
plt.ylabel('temperature')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(0,30)
#ax.spines['left'].set_visible(False)


# In[41]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]
plt.plot(days, temp_min,"b")
plt.plot(days, temp_min,"ob",label = "Min")
plt.plot(days, temp_max,"r")
plt.plot(days, temp_max,"or",label = "Max")
plt.xlabel('days')
plt.ylabel('temperature')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(loc = "best")
#For seperating and naming the ticks in the form of weeks
plt.xticks(range(0,28,4))
plt.show()


# In[32]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]
plt.plot(days, temp_min,"b")
plt.plot(days, temp_min,"ob")
plt.plot(days, temp_max,"r")
plt.plot(days, temp_max,"or")
plt.xlabel('days')
plt.ylabel('temperature')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#For naming the ticks on the x axis as W1 and so on
plt.xticks(range(0,28,4),['W1','W2','W3','W4','W5','W6','W7'])
plt.show()


# In[35]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]
plt.plot(days, temp_min,"b")
plt.plot(days, temp_min,"ob")
plt.plot(days, temp_max,"r")
plt.plot(days, temp_max,"or")
plt.xlabel('days')
plt.ylabel('temperature')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#For naming the ticks on the x axis as W1 and so on
plt.xticks(range(0,28,4),['W1','W2','W3','W4','W5','W6','W7'])

#For removing the axis
plt.axis("off")
plt.show()


# In[40]:


x = [10,20,30,40,50,60,70]
y = [11,25,33,48,55,66,77]
plt.axes().set_aspect('equal')
plt.plot(x,y)


# In[43]:


x = [10,20,30,40,50,60,70]
y = [11,25,33,48,55,66,77]
plt.xlabel('X')
plt.ylabel('Y')
plt.axes().invert_xaxis()
plt.axes().invert_yaxis()
plt.plot(x,y)


# In[47]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]
plt.plot(days, temp_min,"b")
plt.plot(days, temp_min,"ob",label = "Min")
plt.plot(days, temp_max,"r")
plt.plot(days, temp_max,"or",label = "Max")
plt.xlabel('days')
plt.ylabel('temperature')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#for positioning of legend

plt.legend(loc = "best")

#For Annotation
plt.annotate("Temp difference \n across months",xy= (6,25))

#For seperating and naming the ticks in the form of weeks

plt.xticks(range(0,28,4))
plt.show()


# In[49]:


temp_min = [23,25,30,31,29,35,25,39]
temp_max = [29,28,35,36,33,37,28,41]


# In[57]:


plt.figure(figsize=[14,12])

sub1 = plt.subplot(2,2,1)
sub1.set_xticks(())
sub1.set_yticks(())

sub2 = plt.subplot(2,2,2)


sub3 = plt.subplot(2,2,3)
sub3.set_xticks(())
sub3.set_yticks(())

sub4 = plt.subplot(2,2,4)

sub1.plot(days, temp_min,"or")
sub2.plot(days, temp_min,"r")

sub3.plot(days, temp_max,"ob")
sub4.plot(days, temp_max,"b")


# In[58]:


import matplotlib.gridspec as gridspec

gridspec.GridSpec(3,3)
plt.subplot2grid((2,2),(0,0), colspan = 1, rowspan = 1)
plt.plot(days, temp_min)

plt.subplot2grid((2,2),(0,1), colspan = 1, rowspan = 1)
plt.plot(days, temp_max)

plt.subplot2grid((2,2),(1,0), colspan = 2, rowspan = 1)
plt.plot(days, temp_min)


# In[59]:


pl = plt.scatter(temp, days, c= days)
plt.colorbar(pl)


# In[8]:


#Histogram and Countours and HeatMaps

gauss = np.random.normal(size=1000)


# In[18]:


import matplotlib.pyplot as plt
plt.plot(gauss)


# In[23]:


plt.hist(gauss, bins = 20, color = "g", edgecolor = "r")


# In[32]:


#Contours and heatmaps plot 3D into 2D
import matplotlib
xlist = np.linspace(-1,1,30)
ylist = np.linspace(-1,1,30)


# In[31]:


X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X*X+Y*Y)
cp = plt.contour(X,Y,Z)
plt.clabel(cp, fontsize=15)


# In[33]:


X,Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X*X+Y*Y)
cp = plt.contourf(X,Y,Z)


# In[40]:


#HEatMAPS
veg = ["cucumber","tomato","Lettuce","Asparagus","Potato","Okra"]
farms = ["Joe","Smith","Bob","Marley","Elina","Gilbert"]


# In[68]:


harvest = np.array([[0.8,2.4,1.2,3.4,3.9,6.0],[1.1,1.3,1.5,1.7,1.9,2.0],
                    [2.1,2.4,2.6,2.8,3.0,3.3],[3.5,3.2,3.7,3.9,3.0,4.1],
                    [1.5,2.0,2.5,3.0,1.6,0.9],[3.5,3.2,3.7,1.1,1.3,1.5]])
fig, ax = plt.subplots()
im = ax.imshow(harvest)

ax.set_xticks(np.arange(len(farms)))
ax.set_yticks(np.arange(len(veg)))

ax.set_xticklabels(farms)
ax.set_yticklabels(veg)

plt.setp(ax.get_xticklabels(),rotation=45, rotation_mode= "anchor", ha="right")
plt.savefig("Fig1 of ML.jpg")


# In[50]:



##### MATPLOTLIB VS OPENCV

import cv2
print(cv2.__version__)


# In[53]:


img = cv2.imread('E:\D Drive\desktop files\My Pics\Gautam2017.jpg')
img


# In[73]:


img2 = img*2
print(img2)
cv2.imshow('frame', img2)
cv2.waitKey(0)


# In[74]:


cropped_img = img[200:350, 500:700]


# In[59]:


cv2.imshow('frame', img)
cv2.waitKey(0)


# In[60]:


plt.imshow(img)


# In[63]:


rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
plt.imshow(rgb)


# In[70]:


gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[71]:


gray1 = cv2.resize(gray, None, fx = 0.5, fy = 0.5)
gray = cv2.resize(gray,(100,100))
plt.imsave("MyImgGautam.jpg", gray)


# In[ ]:





# In[ ]:




