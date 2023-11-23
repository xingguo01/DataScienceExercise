import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
     import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)

Auto = pd.read_csv('Auto.csv')
print(Auto.columns)
X=pd.DataFrame({'intercept': np.ones(Auto.shape[0]),
                  'horsepower': Auto['horsepower']})
print(X[:4])
y=Auto['mpg']
model=sm.OLS(y,X)
results=model.fit()
print(summarize(results))

design = MS(['horsepower'])
design = design.fit(Auto)
X = design.transform(Auto)

new_df = pd.DataFrame({'horsepower':[98]})
newX = design.transform(new_df)
print(newX)
new_predictions = results.get_prediction(newX);
print(new_predictions.predicted_mean)
print(new_predictions.conf_int(alpha=0.05))
print(new_predictions.conf_int(obs=True, alpha=0.05))

###Question 9###

pd.plotting.scatter_matrix(Auto)
plt.savefig(r"Auto_scatter.pdf")
print(pd.DataFrame.corr(Auto))
terms=Auto.columns.drop(['mpg','name'])
print(terms)
X=MS(terms).fit_transform(Auto)
print(X)
model=sm.OLS(y,X)
results=model.fit()
print(summarize(results))

ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');
plt.savefig(r"Outlier.pdf")
##predictions=model.predict(X)
##table=anova_lm(model.fit())
infl = results.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)
plt.savefig(r"Leverage.pdf")

var_interactions=list(terms)+[('displacement','acceleration'),('weight','acceleration'),('displacement','weight')]
X=MS(var_interactions).fit_transform(Auto)
print(X)
model1=sm.OLS(y,X)
results1=model1.fit()
print(summarize(results1))
