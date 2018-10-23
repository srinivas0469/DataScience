# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\\DataScienceTraining\\Training1\\Simple_Linear_Regression\\Simple_Linear_Regression\\Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.coef_
regressor.intercept_
# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_error = y_test - y_pred
mse = (y_error**2).mean()
rmse = np.sqrt(mse)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


###Summary

import statsmodels.api as sma
X_train = sma.add_constant(X_train) ## let's add an intercept (beta_0) to our model
X_test = sma.add_constant(X_test) 

import statsmodels.formula.api as sm
lm2 = sm.OLS(y_train,X_train).fit()
lm2.summary()
y_pred2 = lm2.predict(X_test) 


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()