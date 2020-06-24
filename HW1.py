# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read file
df=pd.read_csv("C:\\Users\\Caroline\\Documents\\4thSem\\Assignments\\Homework1\\RidingMowers2.csv",sep=',')

#Assign variables
x=df['Income']
y=df['LotSize']
z=df['Ownership']

color_labels = z.unique()

# List of RGB triplets
rgb_values = sns.color_palette("Set2", 8)

# Map label to RGB
color_map = dict(zip(color_labels, rgb_values))


#Plot the scatterplot
plt.scatter(x, y, c=z.map(color_map))
plt.xlabel('Income')
plt.ylabel('LotSize')
plt.legend(z,loc="best")
plt.title('LotSize v/s Income ')


---------------------------
#Read file
df1=pd.read_csv("C:\\Users\\Caroline\\Documents\\4thSem\\Assignments\\Homework1\\ApplianceShipments.csv",sep=',')
df1
#Split first column in Quarter and Year
df1[['Qtr','Year']] = df1.Quarter.str.split("-",expand=True)
df1
#Assign variables
Shipment=df1['Shipments']
Qtr=df1['Qtr']
Year=df1['Year']

#Pivot the data
pv = pd.pivot_table(df1, values='Shipments', index=['Year'],columns=['Qtr'], aggfunc=np.sum)
pv

#Plot the chart
pv.plot()

