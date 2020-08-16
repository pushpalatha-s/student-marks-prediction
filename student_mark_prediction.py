# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 17:25:34 2020

@author: admin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('F:\Latha\MachineLearning\Machine Learning\ML\Practical\Student_Mark_Prediction\student_mark.csv')
df.head()
df.tail()
df.shape
df.info()
df.describe()


#Visualize
plt.scatter(x =df.study_hours, y = df.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Students marks")
plt.title("Scatter Plot of Students Study Hours vs Students marks")
plt.show()

#Prepare the data for ML algorithms
#Data Cleaning
#Find missing values
df.isnull().sum()

df.mean()
df2 = df.fillna(df.mean())
df2.isnull().sum()

df2.head()

#Divide dataset into dependent and independent variable and training and testing
#split dataset
X = df2.drop("student_marks", axis = "columns")
y = df2.drop("study_hours", axis = "columns")

print("Shape of X = ", X.shape)
print("Shape of YY = ", y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=0)
print("Shape of X_train = ", X_train.shape)
print("Shape of y_train = ", y_train.shape)
print("Shape of X_test = ", X_test.shape)
print("Shape of y_test = ", y_test.shape)

#Apply Machine learning Models
#Select a model and train
#y = m * x + c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

#Prediction result
y_pred  = lr.predict(X_test)
y_pred

pd.DataFrame(np.c_[X_test, y_test, y_pred], columns = ["study_hours", "student_marks_original","student_marks_predicted"])

#Predict new result and check accuracy
lr.predict([[6]])[0][0].round(2)
lr.predict([[1]])[0][0].round(2)

lr.score(X_test,y_test)
plt.scatter(X_train,y_train)
plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color = "r")

#Save Student Percentage Prediction Model for deployment
import joblib
joblib.dump(lr, "F:\Latha\MachineLearning\Machine Learning\ML\Practical\Student_Mark_Prediction\Students_mark_predictor_model.pkl")

model = joblib.load("Students_mark_predictor_model.pkl")
model.predict([[5]])[0][0].round(2)