#!/usr/bin/env python
# coding: utf-8

# In[82]:


# import necessary modules
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import gc

import seaborn as sns
sns.set(style = 'whitegrid', color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')

#For statistical tests
import scipy.stats as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import xgboost as xgb
import operator
import os
import random


# In[5]:


#Check the directory path
os.getcwd()

#Set the directory path
os.chdir('C:\\Users\\Caroline\\Documents\\4thSem\\Project\\Data')


# In[140]:


# Filter Row Numbers

n = 125497040 #number of records in file (excludes header)
s = 1000 #desired sample size
select = sorted(random.sample(range(1,n+1),s))
skip  = tuple(set(range(1,n+1)) - set(select))


# In[165]:


# read data sets (large ones)
df_train = pd.read_csv("train.csv",skiprows=skip)
df_test = pd.read_csv("test.csv")


# In[16]:


#read the data sets (small ones)
df_stores = pd.read_csv("stores.csv")
df_items = pd.read_csv("items.csv")
df_trans = pd.read_csv("transactions.csv")
df_oil = pd.read_csv("oil.csv")
df_holiday = pd.read_csv("holidays_events.csv")


# In[9]:


#Count the number of stores
df_stores["store_nbr"].unique().shape[0]


# In[10]:


#Count the number of items
df_items["item_nbr"].unique().shape[0]


# In[27]:


##DATA PREPROCESSING : check for missing data
#Oil data set
oil_nan = (df_oil.isnull().sum() / df_oil.shape[0]) * 100
oil_nan


# In[28]:


#Store dataset
store_nan = (df_stores.isnull().sum() / df_stores.shape[0]) * 100
store_nan


# In[29]:


#Items dataset
item_nan = (df_items.isnull().sum() / df_items.shape[0]) * 100
item_nan


# In[150]:


#Training dataset
df_train_nan = (df_train.isnull().sum() / df_train.shape[0]) * 100
df_train_nan


# In[167]:


#Replacing Nan of "on promotion" with 2 to indicate the items have unknown status on promotion
df_train['onpromotion'] = df_train['onpromotion'].fillna(2)
df_train['onpromotion'] = df_train['onpromotion'].replace(True,1)
df_train['onpromotion'] = df_train['onpromotion'].replace(False,0)
(df_train['onpromotion'].unique())


# In[168]:


#Check if missing values have been imputed
df_train.isnull().sum()


# In[169]:


#Testing dataset
df_test_nan = (df_test.isnull().sum() / df_test.shape[0]) * 100
df_test_nan


# In[170]:


#Holiday dataset
df_holiday_nan = (df_holiday.isnull().sum() / df_holiday.shape[0]) * 100
df_holiday_nan


# In[171]:


##JOINING DATASETS
#Join the train,store,holiday and oil datasets
train = pd.merge(df_train, df_stores, on= "store_nbr")
train = pd.merge(train, df_items, on= "item_nbr")
train = pd.merge(train, df_holiday, on="date")
train = pd.merge(train, df_oil, on ="date")


# In[172]:


#Check the data in the combined dataset
train.head()
train.to_csv('C:\\Users\\Caroline\\Documents\\4thSem\\Project\\Data\\Combined.csv', index=False)


# In[173]:


#Join test and other datasets
test = pd.merge(df_test, df_stores, on= "store_nbr")
test = pd.merge(test, df_items, on= "item_nbr")
test = pd.merge(test, df_holiday, on="date")
test = pd.merge(test, df_oil, on ="date")


# In[174]:


#Check the data
test.head()


# In[32]:





# In[33]:


#Replacing Unknown oil price with 0 for now
train['dcoilwtico'] = train['dcoilwtico'].fillna(0)


# In[175]:


#Splitting the year field in the training and testing dataset
train['Year']  = train['date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['date'].apply(lambda x: int(str(x)[5:7]))
train['date']  = train['date'].apply(lambda x: (str(x)[8:]))


test['Year']  = test['date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['date'].apply(lambda x: int(str(x)[5:7]))
test['date']  = test['date'].apply(lambda x: (str(x)[8:]))


# In[35]:


#Modifying datatypes
train_items1['date'] = pd.to_datetime(train_items1['date'], format='%Y-%m-%d')
train_items1['day_item_purchased'] = train_items1['date'].dt.day
train_items1['month_item_purchased'] =train_items1['date'].dt.month
train_items1['quarter_item_purchased'] = train_items1['date'].dt.quarter
train_items1['year_item_purchased'] = train_items1['date'].dt.year
train_items1.drop('date', axis=1, inplace=True)

train_items2['date'] = pd.to_datetime(train_items2['date'], format='%Y-%m-%d')
train_items2['day_item_purchased'] = train_items2['date'].dt.day
train_items2['month_item_purchased'] =train_items2['date'].dt.month
train_items2['quarter_item_purchased'] = train_items2['date'].dt.quarter
train_items2['year_item_purchased'] = train_items2['date'].dt.year
train_items2.drop('date', axis=1, inplace=True)


# In[36]:


train_items1.loc[(train_items1.unit_sales<0),'unit_sales'] = 1 
train_items1['unit_sales'] =  train_items1['unit_sales'].apply(pd.np.log1p) 

train_items1['family'] = train_items1['family'].astype('category')
train_items1['onpromotion'] = train_items1['onpromotion'].astype('category')
train_items1['perishable'] = train_items1['perishable'].astype('category')
cat_columns = train_items1.select_dtypes(['category']).columns
train_items1[cat_columns] = train_items1[cat_columns].apply(lambda x: x.cat.codes)

train_items2.loc[(train_items2.unit_sales<0),'unit_sales'] = 1 
train_items2['unit_sales'] =  train_items2['unit_sales'].apply(pd.np.log1p) 

train_items2['family'] = train_items2['family'].astype('category')
train_items2['onpromotion'] = train_items2['onpromotion'].astype('category')
train_items2['perishable'] = train_items2['perishable'].astype('category')
cat_columns = train_items2.select_dtypes(['category']).columns
train_items2[cat_columns] = train_items2[cat_columns].apply(lambda x: x.cat.codes)


# In[43]:


##EXPLORATORY DATA ANALYSIS
#Plot sales with promotion
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.barplot(x='onpromotion', y='unit_sales', data=train, ax=axis1)


# In[40]:


#Plotting Sales per Store Type 
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='type_x', y='unit_sales', data=train, ax=axis1)


# In[45]:


##Plotting Stores in Cities and states
#Which city has the most stores?
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.countplot(x=df_stores['city'], data=df_stores, ax=axis1)


# In[46]:


#Which state has the most stores?
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.countplot(x=df_stores['state'], data=df_stores, ax=axis1)


# In[47]:


#Plotting sales and promotion
average_sales = train.groupby('date')["unit_sales"].mean()
average_promo = train.groupby('date')["onpromotion"].mean()

fig, (axis1, axis2) = plt.subplots(2,1,figsize=(15,4))
results["Linear"]=train_get_score(linear_model.LinearRegression())
ax1 = average_sales.plot(legend=True,ax=axis1,marker='o',title="Average Sales")
ax2 = average_promo.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Average Promo")


# In[48]:


##HYPOTHESIS TESTING : Chi square test
# Question 1 - Is there any statistically significant relation between Store Type and Cluster of the stores ?
# Null Hypothesis H0 = Store Type (a, b, c, d, e) and Cluster (1 to 17) are independent from each other.
# Alternative Hypothesis HA = Store Tpe and cluster are not independent of each other. There is a relationship between them.
ct = pd.crosstab(df_stores['type'], df_stores['cluster'])
ct.plot.bar(figsize = (15, 6), stacked=True)
plt.legend(title='cluster vs Type')
plt.show()


# In[49]:


st.chi2_contingency(ct)


# In[ ]:


#Interpretation of Result:
#The p-value is much lower than 0.05.
#There is strong evidence that the null hypothesis is False.
#We reject the null hypothesis and conclude that there is a statistically significant correlation between the Store Type and cluster of the stores.


# In[50]:


## HYPOTHESIS TESTING : t-test
# Question 2 - Is there any statistically significant relation between Store Type and Sales of the stores ?
# Null Hypothesis H0 = Promotion and Sales are independent from each other.
# Alternative Hypothesis HA = Promotion and Sales are not independent of each other. There is a relationship between them.
promo_sales = train[train['onpromotion'] == 1.0]['unit_sales']
nopromo_sales = train[train['onpromotion'] == 0.0]['unit_sales']
st.ttest_ind(promo_sales, nopromo_sales, equal_var = False)


# In[ ]:


# Since the p-value is greater than 0.05,we accept the Null hypothesis.i.e.Promotion and Sales are independent of each other.


# In[176]:


## RANDOM FOREST : Predicting sales.
#Splitting the data
X_train = train.drop(['unit_sales', 'description', 'locale_name','locale','city','state','family','type_x','type_y','cluster','class','perishable','transferred', 'dcoilwtico'], axis = 1)
y_train = train['unit_sales']


# In[177]:


#Running the random forest
rf = RandomForestRegressor(n_jobs = -1, n_estimators = 15)
y = rf.fit(X_train, y_train)


# In[178]:


X_train.head()


# In[184]:


test.head()


# In[185]:


X_test = test.drop(['description', 'locale_name','locale','city','state','family','type_x','type_y','cluster','class','perishable','transferred', 'dcoilwtico'], axis = 1)


# In[186]:


y_test = rf.predict(X_test)


# In[91]:


#Outputting the results
result = pd.DataFrame({'id':test.id, 'unit_sales': y_test}).set_index('id')
result = result.sort_index()
result[result.unit_sales < 0] = 0
result.to_csv('C:\\Users\\Caroline\\Documents\\4thSem\\Project\\Data\\RandomForest.csv')


# In[187]:


# Try different numbers of n_estimators - this will take a minute or so
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    rf.set_params(n_estimators=n)
    rf.fit(X_train, y_train)
    scores.append(rf.score(X_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[ ]:




