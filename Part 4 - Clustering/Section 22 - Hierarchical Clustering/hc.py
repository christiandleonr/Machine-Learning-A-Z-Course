# Hierarchical Clustering

# Importing the libraries

import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Importing the data-set

data = pd.read_csv('C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/Part 4 - '
                   'Clustering/Section 22 - Hierarchical Clustering/Mall_Customers.csv')

"""We take all some columns, this is because we need to take the independents variables"""
X = data.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting hierarchical clustering to the mall data-set

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters

"""In this line we plot all cluster starting with 0 and forward to the final number 
X[y_hc == 0 (this is the cluster), 0 (this is the value en x axis)]
X[y_hc == 0 (this is the cluster), 1 (this is the value en y axis)]"""
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.show()