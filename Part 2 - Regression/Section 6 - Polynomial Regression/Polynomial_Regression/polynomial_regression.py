# Polynomial Regression

# Importing the libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

# Importing the data-set

data = pd.read_csv('C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/Part 2 - Regression/Section 6 - '
                   'Polynomial Regression/Polynomial_Regression/Position_Salaries.csv')

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

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the data-set

"""Create a Simple Linear Regression to compare it with the Polynomial Regression"""
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting polynomial Regression to the data-set

"""We use the library PolynomialFeatures to transform our data set on a relation of x1 x1^2 x1^3 etc. 
This library add the column of ones to represent the constant b0 automatically"""
polynomial_features = PolynomialFeatures(degree=4)
X_polynomial = polynomial_features.fit_transform(X)

"""We have to create a new Linear Regression model but now we will to train it with the Polynomial Features results"""
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_polynomial, y)

# Visualizing the Linear Regression results

"""In this line we plot all point make by the relation of X: level and y: Salary"""
plt.scatter(X, y, color='red')

"""In this line we plot the train part and the prediction for the train part 
to create the line that represents the model"""
plt.plot(X, linear_regressor.predict(X), color='blue')

plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results

"""With this we make a more continuous line making the range of values smaller, incrementing of 0.1"""
X_grid = np.arange(min(X), max(X), 0.1)
"""This line is only to reshape the training set into a matrix"""
X_grid = X_grid.reshape((len(X_grid), 1))

"""In this line we plot all point make by the relation of X: level and y: Salary"""
plt.scatter(X, y, color='red')

"""In this line we plot the train part and the prediction for the train part 
to create the line that represents the model"""
plt.plot(X, linear_regressor_2.predict(polynomial_features.fit_transform(X)), color='blue')

plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
"""6.5 it's the level of the employee and te result of the prediction is the salary that he or she
have to had in relation with her or his level according Linear Regression Model"""
linear_regressor.predict([[6.5]])

# Predicting a new result with Polynomial Regression
"""6.5 it's the level of the employee and te result of the prediction is the salary that he or she
have to had in relation with her or his level according Polynomial Regression Model"""
linear_regressor_2.predict(polynomial_features.fit_transform([[6.5]]))