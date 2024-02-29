import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

import matplotlib.pyplot as plt

# set seed 
np.random.seed(2)
X = np.random.normal(size = (50,2))
X[:25,0] += 3 
X[:25,1] -= 4
plt.scatter(X[:,0],X[:,1])
plt.savefig(r'scatter.pdf')
plt.clf()
clustering = KMeans(n_clusters = 2,n_init=20)
clustering.fit(X)
clusters = clustering.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=clusters, s=50)

centers = clustering.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],marker = '*', c='red', s=200, alpha=1)
plt.savefig(r'clustering_K2.pdf')
plt.clf()

fig,ax = plt.subplots(1,2,figsize = (20,8))

# n_init = 1
clustering = KMeans(n_clusters = 3,n_init=1)
clustering.fit(X)

clusters = clustering.predict(X)

ax[0].scatter(X[:, 0], X[:, 1], c=clusters, s=50)

centers = clustering.cluster_centers_
ax[0].scatter(centers[:, 0], centers[:, 1],marker = '*', c='red', s=200, alpha=1)
ax[0].set_title('N_init = 1')

# n_init = 20
clustering = KMeans(n_clusters = 3,n_init=20)
clustering.fit(X)

clusters = clustering.predict(X)

ax[1].scatter(X[:, 0], X[:, 1], c=clusters, s=50)

centers = clustering.cluster_centers_
ax[1].scatter(centers[:, 0], centers[:, 1],marker = '*', c='red', s=200, alpha=1)
ax[1].set_title('N_init = 20')

plt.savefig(r'clustering_K3.pdf')
plt.clf()


# before applying hierachical clutering lets scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print('Mean of first feature before scaling - ',X[:,0].mean())
print('Mean of first feature after scaling - ',X_scaled[:,0].mean())
# after scaling, its pretty close to 0
# dendrogram
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward'))
plt.savefig(r'dendrogram.pdf')
plt.clf()

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward'))
plt.axhline(y = 7,c = 'red',linestyle = '--')
plt.savefig(r'dendrogram_twogroups.pdf')
plt.clf()

#getting predictions 
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
# affinity here is the method used for calucating distance, and through linkage we can speify the different types 
# of linkages
cluster.fit_predict(X_scaled)

# 3 CLUTERSabs
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward'))
plt.axhline(y = 3.4,c = 'red',linestyle = '--')
plt.savefig(r'dendrogram_twogroups.pdf')
plt.clf()
#getting predictions 
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
# affinity here is the method used for calucating distance, and through linkage we can speify the different types 
# of linkages
cluster.fit_predict(X_scaled)
# 3 CLUTERSabs
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward'))
plt.axhline(y = 3.4,c = 'red',linestyle = '--')

#getting predictions 
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
# affinity here is the method used for calucating distance, and through linkage we can speify the different types 
# of linkages
cluster.fit_predict(X_scaled)
plt.savefig(r'dendrogram_threegroups.pdf')
plt.clf()
