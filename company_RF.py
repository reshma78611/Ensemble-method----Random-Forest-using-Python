# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:23:20 2020

@author: HP
"""
import numpy as np
import pandas as pd

company=pd.read_csv('C:/Users/HP/Desktop/assignments submission/Random Forest/Company_Data.csv')
company.columns
company.isna().sum()
company.Sales.median()

#create bins for sales
cut_labels=['low','medium','high']
cut_bins=[-1,5.66,12,17]
company['sales']=pd.cut(company['Sales'],bins=cut_bins,labels=cut_labels)
company.pop('Sales')

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
company['ShelveLoc']=label_encoder.fit_transform(company['ShelveLoc'])
company['Urban']=label_encoder.fit_transform(company['Urban'])
company['US']=label_encoder.fit_transform(company['US'])

array=company.values
X=array[:,0:10]
Y=array[:,10]
#splitting data using K-Fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold=KFold(n_splits=10,random_state=7)

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,max_features=3)
results=cross_val_score(model,X,Y,cv=kfold)
print(results.mean())#73.75
