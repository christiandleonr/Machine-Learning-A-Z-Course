# Data Preprocessing

# Importing the libraries
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing the data-set
data_set = pd.read_csv('Data.csv')
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 3].values

# Taking care of missing data

"""Imputer change the values NaN by the strategy that we define"""
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data

# Encoding the Independent Variable

"""In this part we use LabelEncoder to encode the categorical data"""
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

"""We use OneHotEncoder to create dummy variables"""
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)