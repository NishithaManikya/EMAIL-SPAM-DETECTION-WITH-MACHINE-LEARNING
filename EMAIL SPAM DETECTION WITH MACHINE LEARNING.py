#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


# In[4]:


spam = pd.read_csv('spam.csv')
spam


# In[5]:


spam.shape


# In[6]:


spam.describe()


# In[15]:


spam.info()


# In[8]:


spam.groupby(spam['v1']).size()


# We have a total of 5572 data. There are %83 safe and %17 spam.

# In[10]:


spam.Category = spam.v1.apply(lambda x: 1 if x == 'spam' else 0) 


# We changed the spam values in the Category column with 1 and the raw values with 0.

# In[11]:


spam.head()


# In[12]:


v2 = spam.iloc[:,1] # Messages column


# In[13]:


v2.head()


# In[16]:


ifSpam = spam.iloc[:,0] # Spam column


# In[17]:


ifSpam.head()


# In[24]:


v2_train, v2_test, ifSpam_train, ifSpam_test = train_test_split(v2, ifSpam, test_size=0.25)


# We will use 75% of our dataset for training 

# In[25]:


cv = CountVectorizer()


# With CountVectorizer, text is analyzed and word counts are made and these are converted into vectors.

# In[27]:


features = cv.fit_transform(v2_train)


# In[29]:


features_test = cv.transform(v2_test)


# ## Learning and Predicts
# 

# In[32]:


knModel = KNeighborsClassifier(n_neighbors=1)


# In[33]:


knModel.fit(features, ifSpam_train)


# In[34]:


knPredict = knModel.predict(features_test)


# In[35]:


dtModel = tree.DecisionTreeClassifier()


# In[36]:


dtModel.fit(features, ifSpam_train)


# In[37]:


dtPredict = dtModel.predict(features_test)


# In[38]:


svModel = svm.SVC()


# In[39]:


svModel.fit(features,ifSpam_train)


# In[40]:


svPredict = svModel.predict(features_test)


# In[41]:


rfModel = RandomForestClassifier() 


# In[42]:


rfModel.fit(features, ifSpam_train)


# In[43]:


rfPredict = rfModel.predict(features_test)


# ## Visualization
# 

# In[48]:


from sklearn.metrics import plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve


# In[49]:


def visualization(model):
    predict = model.predict(features_test)
    plot_confusion_matrix(model,features_test,ifSpam_test)
    plot_precision_recall_curve(model,features_test,ifSpam_test)
    plot_roc_curve(model,features_test,ifSpam_test)


# In[ ]:





# ## Support Vector Machine
# 

# In[44]:


print("Number of mislabeled out of a total of %d test entries: %d" % (features_test.shape[0], 
                                                                      (ifSpam_test != svPredict).sum()))


# In[45]:


successRate = 100.0 * f1_score(ifSpam_test, svPredict, average='micro')


# In[46]:


print("The Success Rate was calculated as % : " + str(successRate) + " with Support Vector Machine")


# In[50]:


visualization(svModel)## Support vector model


# In[51]:


visualization(dtModel)## Decision Tree Model


# In[52]:


visualization(rfModel)##Random forest model


# Here in random forest total number of mislabelled entries are 28

# In[ ]:





# In[ ]:




