# SVR

# Importing the libraries

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np

# Importing the data-set

data = pd.read_csv('C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/Part 2 - Regression/Section 7 - '
                   'Support Vector Regression (SVR)/Position_Salaries.csv')

"""We need to take the independent variables"""
"""We take the only independent variable have to be a matrix and for that reason we take it with this 
instruction 1:2 to indicate that it's a matrix"""
X = data.iloc[:, 1:2].values

"""We take the dependent variable"""
y = data.iloc[:, 2].values

# Splitting the data-set into the Training set and Test set

"""We don't have to divide into the training and test sets because we will use all the data to fit the model"""
"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)"""

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)

"""In this part we reshape the array to be able to use the object sc_y"""
y = y.reshape(-1, 1)
y = sc_y.fit_transform(y)

# Fitting the Regression Model to the data-set

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result with Polynomial Regression

"""In this part we use the object sc_X to transform the value that we want to predict and then we use
the sc_y object to do the inverse and get the real values"""

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

# Visualizing the SVR results

"""In this line we plot all points make by the relation of X: level and y: Salary"""
plt.scatter(X, y, color='red')

"""In this line we plot the train part and the prediction for the train part 
to create the line that represents the model"""
plt.plot(X, regressor.predict(X), color='blue')

plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Regression model results (for higher resolution and smoother curve)

"""With this we make a more continuous line making the range of values smaller, incrementing of 0.1"""
X_grid = np.arange(min(X), max(X), 0.1)
"""This line is only to reshape the training set into a matrix"""
X_grid = X_grid.reshape((len(X_grid), 1))

"""In this line we plot all point make by the relation of X: level and y: Salary"""
plt.scatter(X, y, color='red')

"""In this line we plot the train part and the prediction for the train part 
to create the line that represents the model"""
plt.plot(X_grid, regressor.predict(X_grid), color='blue')

plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()