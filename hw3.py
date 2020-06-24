# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:19:20 2020

@author: Caroline
"""
#Import the packages
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score

# Import data set as data frame
df=pd.read_excel("C:\\Users\\Caroline\\Documents\\4thSem\\Assignments\\Homework 3\\BostonHousing.xlsx",sheet_name='Data')

# The last column is whether its a cost neighborhood
# 2nd to last col is the continuous version of this variable
# And is thus excluded (hint: use df.icol)
#Drop columns not required
colsToDrop=['MEDV']
df=df.drop(colsToDrop,axis=1)

# First columns are used as predictors
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Split into training and testing datasets
x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.10,random_state=42,shuffle=True)


# Make and show a tree with 2 levels (hint: set max depth, fit the tree, plot the tree)
model_2 =  DecisionTreeClassifier(max_depth=2,random_state=0)
fig=model_2.fit(x_training_set, y_training_set)
tree.plot_tree(fig)
plt.show()
y_pred = model_2.predict(x_test_set)
print("Accuracy for depth 2 :",metrics.accuracy_score(y_test_set, y_pred))

# Now with three
model_3 =  DecisionTreeClassifier(max_depth=3,random_state=0)
fig=model_3.fit(x_training_set, y_training_set)
tree.plot_tree(fig)
plt.show()
y_pred = model_3.predict(x_test_set)
print("Accuracy for depth 3 :",metrics.accuracy_score(y_test_set, y_pred))

# What is the optimal tree depth?
# Overfit here, but don’t do this in real life
# Use for statement, fit the tree, check accuracy, print accuracy)

depth_list = list(range(2,15))
depth_tuning=np.zeros((len(depth_list),4))
depth_tuning[:,0] = depth_list

for index in range(0, len(depth_list)):
  mytree = DecisionTreeClassifier(max_depth=depth_list[index])
  mytree.fit(x_training_set, y_training_set)
  y_pred = mytree.predict(x_test_set)
  depth_tuning[index,1] = accuracy_score(y_test_set, y_pred)
  depth_tuning[index,2] = precision_score(y_test_set, y_pred)
  depth_tuning[index,3] = recall_score(y_test_set, y_pred)

# Name the columns and print the array as pandas DataFrame
col_names = ['Max_Depth','Accuracy','Precision','Recall']
print(pd.DataFrame(depth_tuning, columns=col_names))

