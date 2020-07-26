import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 


# SINCE WE'RE CREATING OUR OWN DATASET, WE NEED SET UP RANDOM SEED
np.random.seed(0)



# making random clusters of points by using the make_blobs class. The make_blobs class can take in many inputs, but we will be using these specific ones.

# Input

# n_samples: The total number of points equally divided among clusters.
# Value will be: 5000
# centers: The number of centers to generate, or the fixed center locations.
# Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
# cluster_std: The standard deviation of the clusters.
# Value will be: 0.9
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)


# SETTING UP K-MMEANS CLUSTERING
# init: Initialization method of the centroids.
# Value will be: "k-means++"
# k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
# Value will be: 4 (since we have 4 centers)
# n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
# Value will be: 12
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# TRAIN THE MODEL
k_means.fit(X)

# GETTING LABELS FOR DATAPOINTS
k_means_labels = k_means.labels_
k_means_labels


# GET CLUSTER CENTERS
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


# CREATING VISUAL PLOT OF VARIOUS CLUSTERS AND THEIR CENTERS
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


'''----------------------------------------------------------------------------------------------------'''


# DOWNLOADING THE DATASET
!wget -O Cust_Segmentation.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv


# READ THE DATA 
import pandas as pd
cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()

# DROP UNNECESSARY DATA 
df = cust_df.drop('Address', axis=1)
df.head()


# NORMALIZING THE DATA
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet



# THEN TRAIN THE MODEL USING KMEANS
# GET THE LABELS
# ASSIGN THE LABELS TO DATA 
df["Clus_km"] = labels
df.head(5)


# NOW CHECK CENTROID VALUES USING GROUPBY ON THE LABELS
df.groupby('Clus_km').mean()


# LOOK AT CUSTOMERS BASED ON AGE AND INCOME
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()


