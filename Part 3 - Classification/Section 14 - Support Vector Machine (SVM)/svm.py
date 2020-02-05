# Support Vector Machine (SVM)

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import numpy as np

# Importing the data-set

data = pd.read_csv('C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/Part 3 - '
                   'Classification/Section 14 - Support Vector Machine (SVM)/Social_Network_Ads.csv')

"""We take all columns except the last one, this is because we need to take the three independent variables"""
X = data.iloc[:, [2, 3]].values

"""We take the dependent variable"""
y = data.iloc[:, 4].values

# Splitting the data-set into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Classifier to the Training set

classifier = SVC(kernel='sigmoid', random_state=0)
classifier.fit(X_train, y_train)

# predicting the Test set Results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

"""This give to us a array were we can find the matches between the y_test and y_pred"""
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set Results

X_set, y_set = X_train, y_train
"""This line create local variables that are used to plot the minimum 
to the maximum to each independent variable with a certain step"""
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
"""This line make the contour and separate the categories"""
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
"""This line plot all the points of the training set"""
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set Results

X_set, y_set = X_test, y_test
"""This line create local variables that are used to plot the minimum 
to the maximum to each independent variable with a certain step"""
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
"""This line make the contour and separate the categories"""
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
"""This line plot all the points of the training set"""
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()