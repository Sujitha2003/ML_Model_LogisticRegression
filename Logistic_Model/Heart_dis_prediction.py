# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 20:50:31 2022

@author: Suji

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle


df = pd.read_csv(r"C:\Users\Pavi\OneDrive\Desktop\Logistic_Model\framingham.csv")
df.head()

#preprocessing
df.isnull().sum()

df.shape

df.drop(['male','education','sysBP','diaBP'],axis = 1,inplace = True)

df.head()

df["cigsPerDay"].fillna(df["cigsPerDay"].mode()[0],inplace = True)

Null_list = ['BPMeds','totChol','BMI','heartRate','glucose']
for i in Null_list:
    df[i].fillna(df[i].mode()[0],inplace = True)

df.isnull().sum()

df.info()

df.describe()

# Train and Test the model
X = df.drop('TenYearCHD',axis = 1)
y = df['TenYearCHD']
X_train, X_test,y_train, y_test = train_test_split(X,y,random_state=40,test_size=0.30)


#Use Logistic Algorithm
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#predict
pred = logmodel.predict(X_test)

print(pred)
confusionMatrix = confusion_matrix(y_test,pred)
confusionMatrix

accuracy = accuracy_score(y_test,pred)
print(accuracy)

# Dump as pickle file
with open('model_log_pkl','wb') as model_log_pkl:
    pickle.dump(logmodel,model_log_pkl)

