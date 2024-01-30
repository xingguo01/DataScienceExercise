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
#from sklearn.datasets import fetch_california_housing                                                                                                                                
oj_df = pd.read_csv('../oj.csv')

#oj_df = oj_df.drop(oj_df.columns[0], axis=1)
display(oj_df.head())


# Index for Training set of 800
np.random.seed(1)
train_sample = np.random.choice(np.arange(len(oj_df)), size=800, replace=False)
train = np.asarray([(i in train_sample) for i in oj_df.index])

oj_df['Purchase'] = oj_df['Purchase'].map({'CH' : 1, 'MM': 0})
oj_df['Store7'] = oj_df['Store7'].map({'Yes' : 1, 'No': 0})
display(oj_df.head())

f = 'C(Purchase) ~ ' + ' + '.join(oj_df.columns.drop(['Purchase']))
y, X = pt.dmatrices(f, oj_df)
y = y[:, 0]

# Fit Sklearns tree classifier
clf = tree.DecisionTreeClassifier().fit(X[train], y[train])

print('training accuracy: {}'.format(np.around(clf.score(X[train], y[train]), 3)))
print('leaf nodes: 6')

# Visualise the tree with GraphViz
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=X.design_info.column_names, 
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data) 
#display(HTML(graph._repr_svg_()))

# Here's the confusion matrix
print(confusion_matrix(y[~train], clf.predict(X[~train])))

test_err = 1 - clf.score(X[~train], y[~train])
print('\ntest error rate: {}'.format(np.around(test_err, 3)))

# Estimate optimal tree with cross validation on training set

cv_folds = 10

results = []
for mln in np.arange(2, 50):
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=mln)
    score = cross_val_score(clf, X[train], y[train], cv=cv_folds)
    results += [[mln, np.mean(score)]]


plt.figure(figsize=(10,5))
plot_df = pd.DataFrame(np.asarray(results), columns=['Leaves', 'Test']).set_index('Leaves')
sns.lineplot(data=plot_df);
plt.ylabel('accuracy')
plt.show();

display(HTML('<h4>Optimal tree size:</h4>'))
display(plot_df[plot_df['Test'] == plot_df['Test'].max()])


# Determine actual optimal tree using test set.

results = []
for mln in np.arange(2, 50):
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=mln).fit(X[train], y[train])

    accuracy_train = clf.score(X[train], y[train])  
    accuracy_test = clf.score(X[~train], y[~train])  
    results += [[mln, accuracy_train, accuracy_test]]

plt.figure(figsize=(10,5))
plot_df = pd.DataFrame(np.asarray(results), columns=['Leaves', 'Train', 'Test']).set_index('Leaves')
sns.lineplot(data=plot_df);
plt.ylabel('accuracy')
plt.show();

display(HTML('<h4>Optimal tree size:</h4>'))
display(plot_df[plot_df['Test'] == plot_df['Test'].max()])

clf_unpruned = tree.DecisionTreeClassifier().fit(X[train], y[train])
clf_pruned   = tree.DecisionTreeClassifier(max_leaf_nodes=8).fit(X[train], y[train])

scores = [['unpruned_train', 1 - clf_unpruned.score(X[train], y[train])],
          ['pruned_train', 1 - clf_pruned.score(X[train], y[train])],
          ['unpruned_test', 1 - clf_unpruned.score(X[~train], y[~train])],
          ['pruned_test', 1 - clf_pruned.score(X[~train], y[~train])]]

plot_df = pd.DataFrame(scores, columns=['test', 'error rate'])

plt.figure(figsize=(10, 6))
sns.barplot(x='test', y='error rate', data=plot_df)
plt.show();

display(plot_df)


