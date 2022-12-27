#  mamba install -c anaconda seaborn
#  mamba install -c conda-forge pyodide-py

import pandas
import matplotlib
import scipy
import seaborn

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
# Pyodide is a port of CPython to WebAssembly/Emscripten.
# used for download
from pyodide.http import pyfetch

#%matplotlib inline

# importing os module  
import os 
    
# path 
output = 'output'
    
# Create the directory 
# 'GeeksForGeeks' in 
# '/home / User / Documents' 
try: 
    os.mkdir(output) 
except OSError as error: 
    print(error)  


# *args (Non-Keyword Arguments)
# **kwargs (Keyword Arguments)

# ?????????????????
def warn(*args, **kwargs):
    pass
import warnings
# ?????????????????
warnings.warn = warn
# ?????????????????
warnings.filterwarnings('ignore')

# 
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f: # write binary mode
            f.write(await response.bytes())

# First we need to set a random seed. Use numpy's random.seed() function, where the seed will be set to 0.
np.random.seed(0)

"""Next we will be making random clusters of points by using the make_blobs class. The make_blobs class can take in many inputs, but we will be using these specific ones.

Input

n_samples: The total number of points equally divided among clusters.
Value will be: 5000
centers: The number of centers to generate, or the fixed center locations.
Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
cluster_std: The standard deviation of the clusters.
Value will be: 0.9

Output
X: Array of shape [n_samples, n_features]. (Feature Matrix)
The generated samples.
y: Array of shape [n_samples]. (Response Vector)
The integer labels for cluster membership of each sample.
"""

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.savefig(output + '/k-means_1a.png')

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
# k-means++ - "smart way"
# Clusters ... helping it out
# init = number of iterations

k_means.fit(X)

k_means_labels = k_means.labels_
k_means_labels

# Labels ??????????????????

max_value = np.max(k_means_labels)
print('Maximum value of the array is',max_value)

len(k_means_labels)

type(k_means)

k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
# ?????????????????????????????????? how does this work???????????????????

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
############# ???????????????????????????????????????????????????????????????????????????????/

    # Create a list of all data points, where the data points that are 
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
plt.savefig(output + '/k-means_1b.png')



# write your code here
k_means3j = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means3j.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3j.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
    my_members = (k_means3j.labels_ == k)
#    cluster_center = k_means_cluster_centers[k] ################# ?????????? where is the previous dot????
    cluster_center = k_means3j.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
# Show the plot
plt.savefig(output + '/k-means_1c.png')


