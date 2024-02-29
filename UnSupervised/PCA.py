import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
data = pd.read_csv(r'../USArrests.csv')
print(data.shape)
data.head()
states = data.iloc[:,0]
#names of the states in alphabatical order
print(states)
X = data.iloc[:,1:]
list(X.columns)
print(X.describe())
# scaling the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled,columns = X.columns)
X_scaled_df.head()
print(X_scaled_df.describe())

# using pca to get principal components 
pca = PCA(n_components=4)
pca.fit(X_scaled)
#loading vectors
pd.DataFrame(pca.components_,columns = X.columns,index = ['PC_1','PC_2','PC_3','PC_4']).T
# getting the points in principal components
pca_components = pca.transform(X_scaled)
print(pca_components.shape)
df = pd.DataFrame(pca_components,columns = ['PC_1','PC_2','PC_3','PC_4'])
print(df.head())


# biplot 
fig, ax = plt.subplots(figsize = (20,12))
ax.grid(True)
ax.scatter(df['PC_1'],df['PC_2'])

names = list(data.iloc[:,0])

for i, txt in enumerate(names):
    ax.annotate(txt, (df['PC_1'][i], df['PC_2'][i]))
    
xvector = pca.components_[0] # see 'prcomp(my_data)$rotation' in R
yvector = pca.components_[1]

xs = pca.transform(X_scaled)[:,0] # see 'prcomp(my_data)$x' in R
ys = pca.transform(X_scaled)[:,1]


## visualize projections
    
## Note: scale values for arrows and text are a bit inelegant as of now,
##       so feel free to play around with them

for i in range(len(xvector)):
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i], yvector[i],
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[i]*1.2, yvector[i]*1.2,
             list(X.columns.values)[i], color='r')  

plt.savefig(r'PC1_PC2.pdf')
plt.clf()
# varianc explained by each pc's
pca.explained_variance_
# variance explained ratio's 
pca.explained_variance_ratio_
pd.Series(pca.explained_variance_ratio_,index = ['PC_1','PC_2','PC_3','PC_4']).plot.bar()
pd.Series(np.cumsum(pca.explained_variance_ratio_),index = ['PC_1','PC_2','PC_3','PC_4']).plot.bar()
