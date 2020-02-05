# K-Means

# Importing the libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Importing the data-set
data = pd.read_csv('C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/Part 4 - '
                   'Clustering/Section 21 - K-Means Clustering/Mall_Customers.csv')

"""We take all some columns, this is because we need to take the independents variables"""
X = data.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters

"""This create a chart with the results of each K-Means inertia"""
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-Means to the mall data-set

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualising th clusters

"""In this line we plot all cluster starting with 0 and forward to the final number 
X[y_kmeans == 0 (this is the cluster), 0 (this is the value in x axis)]
X[y_kmeans == 0 (this is the cluster), 1 (this is the value in y axis)]"""
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.show()