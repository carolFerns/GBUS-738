# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:53:45 2020

@author: Caroline
"""

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix	
from sklearn import metrics
from sklearn.feature_selection import RFE
import scikitplot as skplt 

#Read file
df=pd.read_excel("C:\\Users\\Caroline\\Documents\\4thSem\\Assignments\\Homework 2\\UniversalBank.xlsx",sheet_name='Data',skiprows=3)
df

#Assign variables
PersonalLoan=df['Personal Loan']
Age=df['Age']
Income=df['Income']
ZIPCode=df['ZIP Code']
ID=df['ID']

#Plot the scatterplot
color_labels = PersonalLoan.unique()
rgb_values = sns.color_palette("Set2", 8)
color_map = dict(zip(color_labels, rgb_values))
plt.figure(figsize=(10,10))
plt.scatter(Age, Income, c=PersonalLoan.map(color_map))
plt.xlabel('Age',fontsize=20)
plt.ylabel('Income',fontsize=20)
plt.legend(PersonalLoan,loc="best")
plt.title('Age v/s Income',fontsize=30)

df['Personal Loan'].value_counts()
############################################################################
#LOGISTIC REGRESSION

#Drop columns not required
colsToDrop=['ID','ZIP Code']
df=df.drop(colsToDrop,axis=1)
df
#Assign predictor and Target variables
predictors=df.loc[:,df.columns!='Personal Loan']
X=predictors.values

target=df['Personal Loan']
y=target.values

#Run the regression
lr=LogisticRegression()

model=lr.fit(X,y)

preds=model.predict(X)

#Plot the confusion matrix
cnf_matrix=confusion_matrix(y,preds)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#Metrics
metrics.accuracy_score(y,preds)
print(metrics.classification_report(y,preds,digits=2))

#Cut off probability
y=pd.DataFrame(y)	
pred_log = pd.DataFrame(preds)
predict_proba = pd.DataFrame(lr.predict_proba(X))
predictions = pd.concat([y,pred_log,predict_proba],axis = 1)
predictions.columns = [ 'actual', 'predicted', 'Accepted_0', 'Accepted_1']
predictions.head()
fpr, tpr, threshold = metrics.roc_curve(y,predictions.Accepted_1,drop_intermediate=False)
roc_auc = metrics.auc(fpr, tpr)
cutoff_prob = threshold[(np.abs(tpr - 0.6)).argmin()]
round( float( cutoff_prob ), 2 )


#######################################################################################
##Lift Chart

#Select the top 100 customers with the most probability of accepting the loan
predictions
predictions_top100 = predictions.sort_values('Accepted_1',ascending = False).head(100)
predictions_top100

#Add the decile column
predictions_top100['decile'] = pd.qcut(predictions_top100['Accepted_1'],10,labels=['1','2','3','4','5','6','7','8','9','10'])
predictions_top100.head()


#Add a column indicating those who didn't accept the loan
predictions_top100['predicted_0'] = 1-predictions_top100['predicted']

#Pivot table
df1 = pd.pivot_table(data=predictions_top100,index=['decile'],values=['predicted','predicted_0','Accepted_1'],
                     aggfunc={'predicted':[np.sum],
                              'predicted_0':[np.sum],
                              'Accepted_1' : [np.min,np.max]})
df1.head()


df1.reset_index()

df1['Total_Cust'] = df1['predicted']+df1['predicted_0']



df2 = df1.sort_values(by='amin',ascending=False)


predictions_top100.to_csv("C:\\Users\\Caroline\\Documents\\4thSem\\Assignments\\Homework 2\\p100.csv", sep=',', encoding='utf-8')
############

combined=pd.concat([df,predictions],axis=1,join='inner')
top100=combined.sort_values('Accepted_1',ascending = False).head(100)
top100['Personal Loan'].value_counts()
####################################################################
##Feature selection
##Correlation matrix
corr = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()
#Assign predictor and Target variables
predictors=df.loc[:,df.columns!='Personal Loan']
X=predictors.values

target=df['Personal Loan']
y=target.values

selector=RFE(lr,n_features_to_select=3)
selector=selector.fit(predictors,target)

order=selector.ranking_
order

feature_ranks=[]
for i in order:
    feature_ranks.append(f"{i}.{df.columns[i]}")
feature_ranks   

skplt.metrics.plot_cumulative_lift(y,predict_proba)
x