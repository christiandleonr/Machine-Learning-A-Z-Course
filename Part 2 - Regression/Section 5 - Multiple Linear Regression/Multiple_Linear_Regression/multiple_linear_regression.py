# Multiple Linear Regression

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.formula.api as sm

# Importing the data-set
data = pd.read_csv('50_Startups.csv')

"""We take all columns except the last one, this is because we need to take the three independent variables"""
X = data.iloc[:, :-1].values

"""We take the dependent variable"""
y = data.iloc[:, 4]

# Encoding categorical data
# Encoding the Independent Variable

"""In this part we use LabelEncoder to encode the categorical data"""
label_encoder_X = LabelEncoder()
X[:, -1] = label_encoder_X.fit_transform(X[:, -1])

"""We use OneHotEncoder to create dummy variables
The value in [] is the number of the column that we need to encode"""
one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()

"""Avoiding the Dummy Variable Trap"""
X = X[:, 1:]

# Splitting the data-set into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""
We don't have to apply Feature Scaling, the library take care for us. 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

"""

# Fitting Multiple Linear Regression to the Training ser
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
"""In this part we use the independent variables to predict the dependent variable so, the value that this function 
return is an array with the values of the dependent variable on each case"""
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
"""This add a column of ones to represent the constant b0 in our independent variables data set"""

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
