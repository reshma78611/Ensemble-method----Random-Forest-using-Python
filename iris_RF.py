# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:23:26 2020

@author: HP
"""
import numpy as np
import pandas as pd

iris_data=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/decision tree/iris.csv')
array=iris_data.values
X=array[:,0:4]
Y=array[:,4]

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold=KFold(n_splits=10,random_state=7)

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,max_features=3)
results=cross_val_score(model,X,Y,cv=kfold)
print(results.mean())#94.66
