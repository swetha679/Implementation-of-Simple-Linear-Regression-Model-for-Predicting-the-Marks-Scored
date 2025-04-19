# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: B R SWETHA NIVASINI
RegisterNumber:  212224040345
*/
```

```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```


## Output:
![simple linear regression model for predicting the marks scored](sam.png)

# Head Values
![Screenshot 2025-04-19 134759](https://github.com/user-attachments/assets/c419e645-69ee-409b-a554-b0a2a5f5e5fb)

# Tail Values

![Screenshot 2025-04-19 134854](https://github.com/user-attachments/assets/5fd49646-0d70-4bfa-93cb-628c72c5485e)

# Compare Dataset

![Screenshot 2025-04-19 134936](https://github.com/user-attachments/assets/5f8a1cf6-7684-41d8-9518-92439ea59c62)

# Predication values of X and Y

![Screenshot 2025-04-19 135033](https://github.com/user-attachments/assets/bb4a476f-f574-4a81-8442-95b6d2ac7eda)




# Training set

![image](https://github.com/user-attachments/assets/a33492bc-603d-4ca2-a96f-77d1acafb382)

# Testing Set

![Screenshot 2025-04-19 135209](https://github.com/user-attachments/assets/28b5f2bc-7469-41b6-beb9-2c52c52ef07c)


# MSE,MAE and RMSE
![Screenshot 2025-04-19 135246](https://github.com/user-attachments/assets/a6a33c12-b218-413b-a3d2-2259354a2936)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
