#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import sys

a = range(100)
print(sys.getsizeof(10)*len(a))

b = np.arange(100)
print(b.size)


# In[2]:


import numpy as np
import time
import sys

a = range(1)
print(sys.getsizeof(10)*len(a))

b = np.arange(10)
print(b.size)


# In[3]:


import numpy as np
import time
import sys

a = range(100)
print(sys.getsizeof(1)*len(a))

b = np.arange(1)
print(b.size)


# In[4]:


import numpy as np
import time
import sys

a = range(1)
print(sys.getsizeof(1)*len(a))

b = np.arange(1)
print(b.size)


# In[28]:


import numpy as np
import time
import sys

size = 1000000

A1 = range(size)
A2 = range(size)

B1 = np.arange(size)
B2 = np.arange(size)

start = time.time()
result = [(x,y) for x,y in zip(A1,A2)]
print((time.time()-start)*1000)

start = time.time()
result = B1 + B2
print((time.time()-start)*1000)


# In[9]:


import numpy as np
a = np.array([[12,13,14],[11,22,33]])
print(a.shape)
a.shape = (3,2)
print(a)
b = a.reshape(3,2)
print(b)


# In[16]:


a = np.arange(24)
print(a.ndim)
print(a)


# In[18]:


b = a.reshape(2,4,3)
print(b.ndim)
print(b)


# In[19]:


import numpy as np
x = np.array([1,2,3,4,5,6], dtype = np.int8)
print(x.itemsize)


# In[20]:


import numpy as np
x = np.array([1,2,3,4,5,6], dtype = np.float32)
print(x.itemsize)


# In[21]:


import numpy as np
a = np.arange(20)
s = slice(2,19,2)
print(a[s])


# In[26]:


import numpy as np
a = np.arange(20)
b = a[2:19:3]
print(b)


# In[27]:


import numpy as np
a = np.arange(100)
print (a[5:])
print(a[9:19])


# In[34]:


#Iterating over an array

import numpy as np
a = np.arange(0,60,5)
a = a.reshape(3,4)

print("Original array is: ")
print(a)
print("\n\n")

print("Modified array is: ")
for x in np.nditer(a):
    print(x,end=' ')
    


# In[35]:


import numpy as np
a = np.arange(0,60,5)
a = a.reshape(3,4)

print("Original array is: ")
print(a)
print("\n\n")

for x in np.nditer(a, op_flags = ["readwrite"]):
    x[...] = 2*x
print("Modified array is: ")
print(a)


# In[41]:


import numpy as np
a = np.arange(12).reshape(3,4)

print("Original array is: ",a)
print("\n")

print("After applying ravel function: ",a.ravel())
print("\n")

print("After applying Flatten function: ",a.flatten())
print("\n")

print("After applying Transposed function: ",np.transpose(a))
print("\n")

print("After applying T function: ",a.T)


# In[44]:


#Concatenate function Combining two array

import numpy as np
a = np.array([[1,2],[2,3]])

print("First Array: ")
print(a)
print("\n")
b = np.array([[5,6],[7,8]])

print("Second array: ",b)
print("\n")
print("Joining two arrays along axis 0: ")
print(np.concatenate((a,b)))
print("\n")

print("Stack the two arrays along axis 0: ",np.stack((a,b),0))


# In[49]:


import numpy as np
a = np.array([[1,2,3],[4,5,6]])

print("First array: ",a)
print("\n")

print("Append elements to array: ",np.append(a,[7,8,9]))
print("\n")

print("Insert elements to array: ")
print(np.insert(a,6,[7,8,9]))
print("\n")

print("Delete an elements to array: ")
print(np.delete(a,5))
print("\n")


# In[55]:


#Binary operations

import numpy as np
print("Binary equivalents of 13 and 20: ")
a,b = 13,20
print(bin(13),bin(20))
print("\n")

print("Bitwise AND of 13 and 20: ")
print(np.bitwise_and(13,20))
print("\n")

print("Bitwise OR of 13 and 20: ")
print(np.bitwise_or(13,20))
print("\n")

print("Invert of 13 where dtypt of ndarray is uint8: ")
print(np.invert(np.array([13],dtype = np.uint8)))
print("\n")

print("Invert of 20 where dtype of ndarray is uint8: ")
print(np.invert(np.array([20],dtype = np.uint8)))
print("\n")

print("Invert of 20 where dtype of ndarray is uint16: ")
print(np.invert(np.array([20],dtype = np.uint16)))


# In[64]:


import numpy as np
a = np.array([[1,2,3],[4,5,6],[10,11,15]])

print("Array is: ",a)
print("\n")

print("Max of this array = ",np.amax(a))
print("\n")

print("Minimum of this array = ",np.amin(a))
print("\n")

print("Median of this array is = ",np.median(a))
print("\n")

print("Mean of this array is = ",np.mean(a))
print("\n")


# In[65]:


import numpy as np
a = np.array([1,2,3,4,5,6])

print("Our Array is: ",a)
print("\n")

wts = np.array([6,5,4,3,2,1])

print("Applying average() function again: ")
print(np.average(a,weights = wts))


# In[67]:


import numpy as np
print("Std Dev: ",np.std([1,4,6,8,10,12,14]))
print("\n")
print("Variance: ",np.var([1,4,6,8,10,12,14]))


# In[69]:


import numpy as np
a = np.array([3,7,11,9])

print("Our array is: ",a)
print("\n")

print("Applying sort() function: ",np.sort(a))
print("\n")

print("Applying argsort() to a: ",np.argsort(a))


# In[84]:


import numpy as np
x = np.arange(9).reshape(3,3)

print("Our array is: ",x)
print("\n")

print("Printing elements with Indices of element > 3 ")
y = np.where(x>3)
print(x[y])


# In[85]:


import numpy as np
a = np.arange(9).reshape(3,3)

print("Array is: ",a)
print("\n")

condition = np.mod(a,2) == 0

print("Element-wise value of condition: ",condition)
print("\n")

print("Extract elements using conditions: ")
print(np.extract(condition, a))

