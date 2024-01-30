import numpy as np
import pandas as pd
import patsy as pt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from IPython.display import display, HTML
from ipywidgets import interact
import ipywidgets as widgets
import copy
import warnings
warnings.filterwarnings('ignore')

from sklearn import tree
import graphviz 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_california_housing

#housing = fetch_california_housing()
boston_df = fetch_california_housing()   
boston_df = pd.DataFrame(data=np.c_[boston_df['data'], boston_df['target']], columns= [c for c in boston_df['feature_names']] + ['Price'])

np.random.seed(1)
train = np.random.rand(len(boston_df)) < 0.5

display(boston_df.head())

# Create design and response matrix
f = 'Price ~ ' + ' + '.join(boston_df.columns.drop(['Price']))
y, X = pt.dmatrices(f, boston_df)

max_features = {'p': X.shape[1], 
                'p/2': int(np.around(X.shape[1]/2)),
                '$\sqrt{p}$': int(np.around(np.sqrt(X.shape[1]))),
                '1': 1} 

results = []
for mtry in max_features:
    for tree_count in np.arange(1, 100):
        regr   = RandomForestRegressor(max_features=max_features[mtry], random_state=0, n_estimators=tree_count)
        regr.fit(X[train], y[train])
        y_hat = regr.predict(X[~train])
        
        mse = metrics.mean_squared_error(y[~train], y_hat)
        rmse = np.sqrt(mse)
        results+= [[tree_count, mtry, rmse]]

plt.figure(figsize=(10,10))
sns.lineplot(x='Number of Trees', y='RMSE', hue='mtry', 
             data=pd.DataFrame(results, columns=['Number of Trees', 'mtry', 'RMSE']))

plt.savefig(r'RSME_Btrees.pdf')
plt.clf()
