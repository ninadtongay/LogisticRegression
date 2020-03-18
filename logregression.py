#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:41:51 2020

@author: ninadtongay
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('iris.data')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Predicting
y_pred = classifier.predict(X_test)

#Evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
Total number of input in test set: 30

Total number of Iris-setosa: 8
Total number of Iris-versicolor: 11
Total number of Iris-virginica: 11

Number of Iris-setosa predicted: 8
Number of Iris-versicolor predicted: 8
Number of Iris-virginica predicted: 14

Accuracy: (8+8+11)/30 = 0.9
Misclassification Rate: 3/30 = 0.1
"""