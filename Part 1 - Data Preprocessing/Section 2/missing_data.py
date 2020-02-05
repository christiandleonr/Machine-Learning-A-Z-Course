# Data Preprocessing

# Importing the libraries
import pandas as pd
from sklearn.preprocessing import Imputer

# Importing the data-set
data_set = pd.read_csv('Data.csv')
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 3].values

# Taking care of missing data
"""Imputer change the values NaN by the strategy that we define"""
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
