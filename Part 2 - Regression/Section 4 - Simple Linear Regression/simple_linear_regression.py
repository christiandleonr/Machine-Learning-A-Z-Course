# Simple Linear Regression

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Importing the data-set
data = pd.read_csv('Salary_Data.csv')

"""We take all columns except the last one, this is because we need to take the three independent variables"""
X = data.iloc[:, :-1].values

"""We take the dependent variable"""
y = data.iloc[:, 1].values

# Splitting the data-set into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set Results
y_pred = regressor.predict(X_test)  # Vector o predictions - method Predict return a array of predicted values

# Visualising the Training set results

"""With this we plot the information contains in our X_train and y_train, those are the values with that we train
the simple linear regression model"""
plt.scatter(X_train, y_train, color='red')

"""We use the X_train in y axis and the predictions to the X_train for the x axis
This is the line that describe the relation of the independent variable and the dependent variable"""
plt.plot(X_train, regressor.predict(X_train), color='blue')

# Giving information to our plot
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
"""Now we plot the information contained in our test section to visualize the relation of the line with this 
 information"""
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
