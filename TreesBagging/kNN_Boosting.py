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

caravan_df = pd.read_csv('../Caravan.csv')

# Patsy feature processing
f = 'C(Purchase) ~ ' + ' + '.join(caravan_df.columns.drop(['Purchase']))
y, X = pt.dmatrices(f, caravan_df)
y = y[:, 1]

# Display processed features
display(pd.DataFrame(X, columns=X.design_info.column_names).head())

# Index for Training set of 1000
np.random.seed(1)
train_sample = np.random.choice(np.arange(len(caravan_df)), size=1000, replace=False)
train = np.asarray([(i in train_sample) for i in caravan_df.index])
max_features = 'auto'
tree_count   = 1000
learning_rate = 0.01

model = GradientBoostingClassifier(max_features=max_features, 
                                       random_state=1, 
                                       n_estimators=tree_count,
                                       learning_rate=learning_rate)

model = model.fit(X[train], y[train])
#y_hat_test  = regr.predict(X[~train])

accuracy = model.score(X[~train], y[~train])
print('accuracy: {}%'.format(np.around(accuracy*100, 2)))

# Plot feature by importance in this model

plot_df = pd.DataFrame({'feature': X.design_info.column_names, 'importance': model.feature_importances_})

plt.figure(figsize=(10,20))
sns.barplot(x='importance', y='feature', data=plot_df.sort_values('importance', ascending=False),
            color='b')
plt.xticks(rotation=90);
plt.savefig(r'Caravan_Boosting_importance.pdf')
plt.clf()

max_features = 'auto'
tree_count   = 1000
learning_rate = 0.01

model = GradientBoostingClassifier(max_features=max_features, 
                                       random_state=1, 
                                       n_estimators=tree_count,
                                       learning_rate=learning_rate)

model = model.fit(X[train], y[train])
#y_hat_test  = regr.predict(X[~train])


# Boosting stats
threshold = 0.2
y_hat_proba = model.predict_proba(X[~train])
y_hat = (y_hat_proba[:, 1] > threshold).astype(np.float64)
confusion_mat = confusion_matrix(y[~train], y_hat)

# What fraction of the people predicted to make a purchase do in fact make one?
pos_pred_val = np.around(confusion_mat[:, 1][1] / np.sum(confusion_mat[:, 1]), 5)

display(HTML('<h4>BOOSTING: Confusion matrix</h4>'))
print(confusion_mat)

print('\nPositive Predictive Value: {}'.format(pos_pred_val))
