#!/usr/bin/env python
# coding: utf-8

# # GRIP: THE SPARKS FOUNDATION
# 
# 

# # Data science and business analytics intern
# AUTHOR:PRAJWAL E U
# 
# 
# #TASK1: prediction using supervised machine learning
# 

# # STEPS:
# * IMO

# #by simple linear regression method ,we can predict the percentage of a student based on the number of hours studied.

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #installing the seaborn

# In[3]:


pip install seaborn


# In[4]:


import seaborn as sns


# In[23]:


##importing dataset-
url="http://bit.ly/w-data"
data = pd.read_csv(url)
print("data is successfully imported")
data


# In[7]:


data.describe()


# In[8]:


data.head()


# In[9]:


data.tail()


# In[32]:


import seaborn as sns


# In[26]:


plt.boxplot(data)
plt.show()


# In[13]:


data.info()


# THIS SCATTER PLOT INDICATES POSITIVE LINEAR RELATIONSHIP AS MUCH AS HOURS YOU STUDY MORE THE HIGHER THE SCORE GETS

# In[15]:


plt.xlabel('Hours',fontsize =20)
plt.ylabel('Scores',fontsize =20)
plt.title('Hours studied vs scores',fontsize = 15)
plt.scatter(data.Hours,data.Scores,color ='blue',marker ='+')
plt.show()


# In[28]:


data.corr(method='pearson')


# In[35]:


hours =data['Hours']
scores =data['Scores']


# In[36]:


sns.displot(hours)


# In[37]:


sns.displot(scores)


# In[41]:


X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values
X


# In[42]:


Y


# In[43]:


##PREPARING OF DATA AND SPLITTING OF DATAINTO TRAIN AND TEST FUNCTIONS
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0,test_size = 0.2)


# In[44]:


##splitting the data into 80;20 ratio
print("X train SHAPE=",X_train.shape)


# In[45]:


print("Y TRAIN SHAPE=",Y_train.shape)


# In[46]:


print("Y TEST SHAPE=",Y_test.shape)


# In[47]:


print("X TEST SHAPE=",X_test.shape)


# In[48]:


##Training the data
from sklearn.linear_model import LinearRegression
Linreg =LinearRegression()
Linreg.fit(X_train,Y_train)


# In[53]:


m = Linreg.coef_
c =Linreg.intercept_
ans = m*X+c
plt.scatter(X,Y,color='red',marker ='*')
plt.plot(X,ans);
plt.show()


# In[54]:


Y_pred = Linreg.predict(X_test)
print(Y_pred)


# In[55]:


Y_test


# In[62]:


plt.plot(X_test,Y_pred,color='black')
plt.scatter(X_test,Y_pred,color ='red',marker = "*")
plt.xlabel("Hours",fontsize = 20)
plt.ylabel("Scores",fontsize = 20)
plt.title ('LINEAR REGRESSION',fontsize=24)
plt.show()


# In[59]:


#comparing the model
Y_test1 = list(Y_test)
prediction=list(Y_pred)
df_compare = pd.DataFrame({ 'Actual':Y_test1,'Result':prediction})
df_compare


# In[60]:


##accuracy of the model
from sklearn import metrics
metrics.r2_score(Y_test,Y_pred)


# In[72]:


##root mean square error
C = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
print("root mean square error =",C)


# predicting the score

# In[69]:


Prediction_score = Linreg.predict([[9.25]])
print("predicted score for a student studying 9.25 hours :",Prediction_score)


# # If a student studies 9.25 hours/day he will score 93.69% in exam

# In[ ]:




