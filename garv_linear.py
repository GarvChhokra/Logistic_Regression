#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:30:24 2022

@author: garvchhokra
"""

import os
import pandas as pd
filename = "titanic.csv"
path = "/Users/garvchhokra/Desktop/COMP 237/Assignment4/Exercise#1_garv/"
fullpath = os.path.join(path, filename)
titaninc_garv = pd.read_csv(fullpath)

titaninc_garv.head(3)
titaninc_garv.shape
titaninc_garv.info()
"""Columns that are not useful for the logistical regression that are PassengerId, Name, Ticket, Cabin"""
titaninc_garv.dtypes
titaninc_garv['Sex'].unique()
titaninc_garv['Pclass'].unique()

import matplotlib.pyplot as plt
pd.crosstab(titaninc_garv.Survived,titaninc_garv.Pclass).plot(kind='bar')
plt.title('# of survived versus the passenger class - by Garv')
plt.xlabel('Survived')
plt.ylabel('Passenger class')

pd.crosstab(titaninc_garv.Survived,titaninc_garv.Sex).plot(kind='bar')
plt.title('# of survived versus the Sex - by Garv')
plt.xlabel('Survived')
plt.ylabel('Sex')

pd.plotting.scatter_matrix(titaninc_garv[['Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']])

titaninc_garv_drop = titaninc_garv.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
titaninc_garv_dummies = pd.get_dummies(titaninc_garv_drop,columns=["Sex", "Embarked"])
# titaninc_garv_drop = titaninc_garv.drop(columns=["Sex", "Embarked"])
titaninc_garv = titaninc_garv_dummies

titaninc_garv['Age'].fillna(titaninc_garv['Age'].mean(), inplace = True)
titaninc_garv = titaninc_garv.astype(float)
titaninc_garv.info()

def normalizes(df):
    norm = df.copy()
    for i in df.columns:
        norm[i]= (df[i] - df[i].min())/(df[i].max() - df[i].min())
        return norm
titaninc_garv_norm = normalizes(titaninc_garv)
titaninc_garv_norm.head(2)

titaninc_garv_norm.hist(figsize=(9,10))

X_garv = titaninc_garv_norm[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
Y_garv = titaninc_garv_norm[["Survived"]]
from sklearn.model_selection import train_test_split,cross_val_score
X_train_garv, X_test_garv, Y_train_garv, Y_test_garv = train_test_split(X_garv, Y_garv, test_size=0.3, random_state=37)

from sklearn.linear_model import LogisticRegression
garv_model = LogisticRegression()
garv_model.fit(X_train_garv, Y_train_garv)
print(garv_model.coef_)
import numpy as np
pd.DataFrame(zip(X_train_garv.columns, np.transpose(garv_model.coef_)))

score = cross_val_score(garv_model, X_train_garv, Y_train_garv, cv=10)
print(score)
print(score.mean())

mean_score = []
for i in np.arange(0.10, 0.55, 0.05):
    print (i)
    X_train_garv1, X_test_garv1, Y_train_garv1, Y_test_garv1 = train_test_split(
    X_garv, Y_garv, test_size=i, random_state=37)
    garv_model.fit(X_train_garv1, Y_train_garv1)
    scores = cross_val_score(garv_model, X_train_garv1, Y_train_garv1, cv=10)
    print (scores)
    mean_score.append(scores.mean())
    print(f"Minimum: {scores.min()}  Mean: {scores.mean()}   Maximum: {scores.max()}")
print (mean_score)        
x_train_garv, x_test_garv, y_train_garv, y_test_garv = train_test_split(X_garv, Y_garv, test_size=0.3, random_state=37)
garv_model1 = LogisticRegression()
garv_model1.fit(x_train_garv, y_train_garv)
y_pred_garv = garv_model1.predict_proba(x_test_garv)
print(y_pred_garv)

y_pred_garv_flag = y_pred_garv[:,1] > 0.5
print(y_pred_garv_flag)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(accuracy_score(y_test_garv, y_pred_garv_flag))
print(confusion_matrix(y_test_garv, y_pred_garv_flag))
print(classification_report(y_test_garv, y_pred_garv_flag))

"""doing steps again using thershold value of 0.75"""
y_pred_garv_flag = y_pred_garv[:,1] > 0.75
print(y_pred_garv_flag)

print(accuracy_score(y_test_garv, y_pred_garv_flag))
print(confusion_matrix(y_test_garv, y_pred_garv_flag))
print(classification_report(y_test_garv, y_pred_garv_flag))
