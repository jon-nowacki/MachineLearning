import pandas as pd
import matplotlib
import scipy
import seaborn
import random 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import asyncio
import random 

from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 

# Pyodide is a port of CPython to WebAssembly/Emscripten.
# used for download
from pyodide.http import pyfetch
import requests

output = 'output'
try: 
    os.mkdir(output) 
except OSError as error: 
    print(error)  

"""
URL ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv' 

filename ="Cust_Segmentation.csv"

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f: # write binary mode
            f.write(await response.bytes())

asyncio.run(download(URL, "Cust_Segmentation.csv"))
"""

cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()

# Preprocessing
# "Address" in this dataset is a categorical variable. The k-means algorithm isn't directly applicable to 
# categorical variables because the Euclidean distance function isn't really meaningful for discrete variables. 
# So, let's drop this feature and run clustering.
df = cust_df.drop('Address', axis=1)
df.head()

from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

df["Clus_km"] = labels
df.head(5)

df.groupby('Clus_km').mean()

area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.savefig(output + '/k-means_2a.png')

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.savefig(output + '/k-means_2b.png')