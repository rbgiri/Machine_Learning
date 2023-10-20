#!/usr/bin/env python
# coding: utf-8

# ### Overview
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we target to complete the analysis of what sorts of people were likely to survive.

# https://www.kaggle.com/c/titanic/data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

sns.set(rc={'figure.figsize':(12, 10)})


# ### Loading Dataset

# In[2]:


df = pd.read_csv("titanic_data.csv")
df.head()


# ## Types of Features :
# 
# #### Categorical - Sex, and Embarked.
# #### *Continuous * - Age, Fare.
# #### Discrete - SibSp, Parch.
# #### Alphanumeric - Cabin.

# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[7]:


print(test.shape)
train.shape
      


# In[8]:


train.head()


# In[9]:


train.isna().sum()


# In[10]:


train.info()


# In[11]:


train.describe()


# In[12]:


test.head()


# In[13]:


test.isna().sum()


# In[14]:


test.info()


# In[15]:


test.describe()


# ## Numerical Value Analysis

# In[16]:


plt.figure(figsize=(12,10))
heatmap = sns.heatmap(df[["Survived","SibSp","Parch","Age","Fare"]].corr(), annot=True )


# ### *Conclusion: *
# Only Fare feature seems to have a significative correlation with the survival probability.
# 
# It doesn't mean that the other features are not usefull. Subpopulations in these features can be correlated with the survival. To determine this, we need to explore in detail these features

# ## <font color = "Burgundy">Sex Vs Survived</font>

# In[17]:


sns.barplot(x = "Sex",y = "Survived",data = train)

# Print percentage
print("percentage of women who survived : " , train['Survived'][train['Sex']== 'female'].value_counts(normalize= True)[1]*100)
print("percentage of men who survived : " , train['Survived'][train['Sex']== 'male'].value_counts(normalize= True)[1]*100)


# ## <font color = 'Purple'>Pclass VS Survived</font>
# 

# In[18]:


df["Pclass"].unique()


# In[19]:


sns.barplot(x = "Pclass",y = "Survived",data = train)

print("percentage of Pclass = 1 who survived : " , train['Survived'][train['Pclass']== 1].value_counts(normalize= True)[1]*100)
print("percentage of Pclass = 2 who survived : " , train['Survived'][train['Pclass']== 2].value_counts(normalize= True)[1]*100)
print("percentage of Pclass = 3 who survived : " , train['Survived'][train['Pclass']== 3].value_counts(normalize= True)[1]*100)


# ## <font color = "green">sibsp - Number of siblings / spouses aboard the Titanic </font> 

# In[20]:


df['SibSp'].nunique()


# In[21]:


df['SibSp'].unique()


# In[22]:


bargraph_sibsp = sns.factorplot(x = "SibSp", y = "Survived", data = df, kind = "bar", size = 8)
bargraph_sibsp = bargraph_sibsp.set_ylabels("survival probability")


# It seems that passengers having a lot of siblings/spouses have less chance to survive.
# Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have more chance to survive.

# ## <font color = maroon >Agegroup VS Survived </font>

# In[23]:


train.Age = train.Age.fillna(-0.5)
test.Age = test.Age.fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ["Unknown","Baby","Child","Teenager","Student","Young Adult","Adult","Senior"]

train["Agegroup"] = pd.cut(train["Age"],bins,labels = labels)
test["Agegroup"] = pd.cut(test["Age"],bins,labels = labels)

sns.barplot(x = "Agegroup", y = "Survived", data=train)
plt.show()


# In[24]:


train.columns


# In[25]:


train = train.drop(["Ticket","Cabin","Fare"],axis = 1)
test = test.drop(["Ticket","Cabin","Fare"], axis = 1)


# ## <font color = 'green' >Dealing with Missing dataset </font>

# In[26]:


print("Number of people Embarking in Southampton (S) :")
Southampton = train[train['Embarked'] == 'S'].shape[0]
print(Southampton)

print("Number of people Embarking in Cherbourg (C) :")
Cherbourg = train[train['Embarked'] == 'C'].shape[0]
print(Cherbourg)

print("Number of people Embarking in Queenstown(Q) :")
Queenstown = train[train['Embarked'] == 'Q'].shape[0]
print(Queenstown)


# In[27]:


train = train.fillna({"Embarked" : 'S'})


# In[28]:


train.isna().sum()


# ## <font color = blue>Fill missing values in Agegroup column</font>

# In[29]:


train.head()


# In[30]:


combine = [train,test]


# In[31]:




for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand = False)
    
pd.crosstab(train['Title'],train['Sex'])


# In[32]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Jonkheer','Major','Rev'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    
train[['Title','Survived']].groupby(['Title'],as_index = False).mean()


# In[33]:


title_mapping = {'Mr' : 1,'Miss' : 2,'Mrs' : 3,'Master' : 4,'Royal' : 5,'Rare' : 6}

for dataset in combine :
    
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# In[34]:


Mr_age = train[train['Title'] == 1]['Agegroup'].mode()  #Young Adult
Miss_age = train[train['Title'] == 2]['Agegroup'].mode() #Student
Mrs_age = train[train['Title'] == 3]['Agegroup'].mode()  #Adult
Master_age = train[train['Title'] == 4]['Agegroup'].mode() #Baby
Royal_age = train[train['Title'] == 5]['Agegroup'].mode() #Adult
Rare_age = train[train['Title'] == 6]['Agegroup'].mode() #Adult

age_Title_map = {1 :'Young Adult',2 : 'Student',3 : 'Adult',4 : 'Baby' , 5 : 'Adult',6 : 'Adult'}

for i in range(len(train['Agegroup'])) :
    if train['Agegroup'][i] == 'Unknown' :
        train['Agegroup'][i] == age_Title_map[train['Title'][i]]


for i in range(len(test['Agegroup'])) :
    if test['Agegroup'][i] == 'Unknown' :
        test['Agegroup'][i] == age_Title_map[test['Title'][i]]


# In[35]:


train.Agegroup.unique()


# In[36]:


train.head()


# In[37]:


from sklearn.preprocessing import LabelEncoder
transform = ['Agegroup','Sex','Embarked']

le = LabelEncoder()

for i in transform :
    train[i] = le.fit_transform(train[i])


# In[38]:


train.head()


# In[39]:


for i in transform :
    test[i] = le.fit_transform(test[i])


# In[40]:


test.head()


# ## <font color = orange >Machine Learning Model </font>

# In[41]:


X_train = train.drop(['PassengerId','Name','Survived'],axis = 1 )
Y_train = train['Survived']
X_test = test.drop(['PassengerId','Name'],axis = 1)


# In[43]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,Y_train)
Prediction = lr.predict(X_test)


# In[44]:


Prediction 


# In[46]:


Ids = test['PassengerId']

Output = pd.DataFrame({'PassengerId': Ids, 'Survived' : Prediction})
Output.to_csv('Submission.csv',index = False)


# In[ ]:




