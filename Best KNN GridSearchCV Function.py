#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


from sklearn.neighbors import KNeighborsClassifier


# In[3]:


from sklearn.model_selection import GridSearchCV


# In[4]:


data=np.genfromtxt('trainData.csv',delimiter=',', skip_header=1)


# In[5]:


X2 = data[:,np.arange(2)]
Y2 = data[:,2]


# In[6]:


def bestKNNClassifier(features, classes, paramGrid = {'n_neighbors' :  np.arange(1,10,1)} , cvFolds = 10):
        knn = KNeighborsClassifier()

        X_train, X_test, y_train, y_test = train_test_split(features,classes,test_size=.2, random_state=42)
        grid_search = GridSearchCV( knn, param_grid =  paramGrid, cv = cvFolds, scoring='accuracy')

        grid_search.fit(X_train, y_train)

        bestKNN=grid_search.best_estimator_
        bestNeighborNumber = grid_search.best_estimator_.n_neighbors
        y_pred = grid_search.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        return bestKNN,  bestNeighborNumber


# In[7]:


clf, n = bestKNNClassifier(X2,Y2,paramGrid = {'n_neighbors' :  np.arange(1,21,2)})


# In[8]:


n


# In[9]:


clf


# In[10]:


X2


# In[ ]:





# In[ ]:





# In[ ]:




