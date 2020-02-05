# Data Preprocessing Template

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the data-set
data = pd.read_csv('Salary_Data.csv')

"""We take all columns except the last one, this is because we need to take the three independent variables"""
X = data.iloc[:, :-1].values

"""We take the dependent variable"""
y = data.iloc[:, 1].values

# Splitting the data-set into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
