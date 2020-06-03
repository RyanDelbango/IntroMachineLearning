#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


fishData=np.genfromtxt('fish_data.csv',delimiter=',', skip_header=1) 


# In[5]:


fishX = fishData[:,2:5]


# In[6]:


fishX


# In[7]:


fishY = fishData[:,1]


# In[8]:


fishY


# In[9]:


regFish=sk.linear_model.LinearRegression()
regFish.fit(fishX,fishY)


# In[10]:


regFish.coef_


# In[11]:


regFish.intercept_


# In[13]:


regFish.score(fishX,fishY)


# In[29]:


fishX2 = fishData[:,2:5]


# In[30]:


fishX2


# In[32]:


fishX2 = np.delete(fishX2, 1, axis=1)


# In[33]:


fishX2


# In[34]:


regFish2=sk.linear_model.LinearRegression()
regFish2.fit(fishX2,fishY)


# In[36]:


regFish2.coef_


# In[38]:


regFish2.intercept_


# In[39]:


regFish2.score(fishX2,fishY)


# In[ ]:




