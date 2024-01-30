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
hitters_df = pd.read_csv('../Hitters.csv')
# Drop null observations
hitters_df = hitters_df.dropna()
assert hitters_df.isnull().sum().sum() == 0

f = 'np.log(Salary) ~ ' + ' + '.join(hitters_df.columns.drop(['Salary']))
y, X = pt.dmatrices(f, hitters_df)

pd.DataFrame(X, columns=X.design_info.column_names).head()
display(pd.DataFrame(X, columns=X.design_info.column_names).head())

# Index for Training set of 200
np.random.seed(1)
train_sample = np.random.choice(np.arange(len(hitters_df)), size=200, replace=False)
train = np.asarray([(i in train_sample) for i in hitters_df.index])

# Gradient boosting

max_features = 'auto'
tree_count   = 1000

# np.arange(0.0001, 1.2, 0.01)

results = []
for learning_rate in np.logspace(-10, np.log(1.3), 100): 
    regr   = GradientBoostingRegressor(max_features=max_features, 
                                       random_state=1, 
                                       n_estimators=tree_count,
                                       learning_rate=learning_rate)
    regr = regr.fit(X[train], y[train])
    y_hat_train = regr.predict(X[train])
    y_hat_test  = regr.predict(X[~train])
    
    mse_train = metrics.mean_squared_error(y[train], y_hat_train)
    mse_test  = metrics.mean_squared_error(y[~train], y_hat_test)
    
    results += [[learning_rate, mse_train, mse_test]]

# Plot
df = pd.DataFrame(np.asarray(results), 
                  columns=['learning_rate', 'mse_train', 'mse_test']).set_index('learning_rate')
plt.figure(figsize=(10,10))
ax  = sns.lineplot(data=df)
ax.set_xscale('log')
plt.ylabel('MSE')
plt.show();

# Show best learning rate
display(df[df['mse_test'] == df['mse_test'].min()])

# Naive ols
model = sm.OLS(y[train], X[train]).fit()
y_hat = model.predict(X[~train])

mse_test = metrics.mean_squared_error(y[~train], y_hat)
print('MSE test: {}'.format(np.around(mse_test, 3)))


def lasso_cv(X, y, λ, k):
    """Perform the lasso with 
    k-fold cross validation to return mean MSE scores for each fold"""
    # Split dataset into k-folds
    # Note: np.array_split doesn't raise excpetion is folds are unequal in size
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    MSEs = []
    for f in np.arange(len(X_folds)):
        # Create training and test sets
        X_test  = X_folds[f]
        y_test  = y_folds[f]
        X_train = X.drop(X_folds[f].index)
        y_train = y.drop(y_folds[f].index)
        
        # Fit model
        model = Lasso(alpha=λ, copy_X=True, fit_intercept=True, max_iter=10000,
                      positive=False, precompute=False, random_state=0,
                      selection='cyclic', tol=0.0001, warm_start=False).fit(X_train, y_train)
        
        # Measure MSE
        y_hat = model.predict(X_test)
        #print(y_test)
        MSEs += [metrics.mean_squared_error(y_test, y_hat)]
    return MSEs

X_train = pd.DataFrame(X[train], columns=X.design_info.column_names)
y_train = pd.DataFrame(y[train], columns=['Price'])

#lambdas = np.arange(.000001, 0.01, .0001)

lambdas = np.arange(0.2, 20, .1)

MSEs    = [] 
for l in lambdas:
    MSEs += [np.mean(lasso_cv(X_train, y_train, λ=l, k=10))]

sns.scatterplot(x='λ', y='MSE', data=pd.DataFrame({'λ': lambdas, 'MSE': MSEs}))
plt.show();

# Choose model
lamb = min(zip(MSEs, lambdas))
print('RMSE Train CV: {}\n@Lambda: {}'.format(np.sqrt(lamb[0]), lamb[1]))


# Use chosen model on test set prediction
model = Lasso(alpha=lamb[1], copy_X=True, fit_intercept=True, max_iter=10000,
              positive=False, precompute=False, random_state=0,
              selection='cyclic', tol=0.0001, warm_start=False).fit(X[train], y[train])

y_hat = model.predict(X[~train])

mse = metrics.mean_squared_error(y[~train], y_hat)
rmse = np.sqrt(mse)

print('MSE test: {}'.format(np.around(mse, 3)))
print('RMSE test: {}'.format(np.around(rmse, 3)))

max_features = 'auto'
tree_count   = 1000
learning_rate = 0.00229

regr = GradientBoostingRegressor(max_features=max_features, 
                                       random_state=1, 
                                       n_estimators=tree_count,
                                       learning_rate=learning_rate)

regr = regr.fit(X[train], y[train])
y_hat_test  = regr.predict(X[~train])

mse_test  = metrics.mean_squared_error(y[~train], y_hat_test)
print(mse_test)

# Plot feature by importance in this model

plot_df = pd.DataFrame({'feature': X.design_info.column_names, 'importance': regr.feature_importances_})

plt.figure(figsize=(10,10))
sns.barplot(x='importance', y='feature', data=plot_df.sort_values('importance', ascending=False),
            color='b')
plt.xticks(rotation=90)
# Bagging with 100 trees
# although I'm using RandomForestRegressor algo here this is Bagging because max_features = n_predictors

max_features = X.shape[1]
tree_count   = 1000

regr   = RandomForestRegressor(max_features=max_features, random_state=0, n_estimators=tree_count)
regr.fit(X[train], y[train])
y_hat = regr.predict(X[~train])

mse = metrics.mean_squared_error(y[~train], y_hat)
rmse = np.sqrt(mse)

print('MSE test: {}'.format(np.around(mse, 3)))
print('RMSE test: {}'.format(np.around(rmse, 3)))
