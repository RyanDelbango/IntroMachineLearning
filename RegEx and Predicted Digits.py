#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[2]:


pattern = (r'[0-9a-f]{2}\:[0-9a-f]{2}\:[0-9a-f]{2}\:[0-9a-f]{2}\:[0-9a-f]{2}\:[0-9a-f]{2}')
with open("dump.txt", "r") as f:
    d=f.read()
    mac_addresses= re.findall(pattern, d)


# In[3]:


unique_mac_addresses= set(re.findall(pattern, d))


# In[4]:


unique_mac_addresses


# In[5]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


# In[6]:


data=np.genfromtxt("mnist1000.csv", delimiter=',', skip_header=1)


# In[9]:


labels=data[:,0]
len(labels)


# In[10]:


X=data[:,1:]


# In[16]:


X.shape
X


# In[12]:


plt.figure(figsize=(1, 1))
plt.imshow(X[5].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# In[13]:


pca = PCA(n_components=100)
pca.fit(X)


# In[30]:


Xt = pca.fit_transform(X)
Xt


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(Xt,labels,test_size=.2, random_state=42)


# In[32]:


clf=SVC(gamma='scale')
param_grid = {
    'kernel':['linear', 'rbf', 'poly'],
    'C': [1, 50, 100, 1000]
}
grid_search = GridSearchCV(
    clf, param_grid, cv=10, scoring='accuracy')


# In[33]:


grid_search.fit(X_train, y_train)


# In[34]:


from sklearn import metrics
bestModel=grid_search.best_estimator_
metrics.accuracy_score(y_test, bestModel.predict(X_test))


# In[35]:


unknownDigits=np.genfromtxt("RecognizeDigits.csv", delimiter=',')


# In[36]:


predictedDigits = bestModel.predict(pca.transform(unknownDigits))
predictedDigits


# In[37]:


plt.figure(figsize=(1, 1))
plt.imshow(unknownDigits[0,:].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# In[38]:


plt.figure(figsize=(1, 1))
plt.imshow(unknownDigits[1,:].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# In[39]:


plt.figure(figsize=(1, 1))
plt.imshow(unknownDigits[2,:].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# <h5>Yes the predictions, match the actual digits
