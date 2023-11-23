###Exercise###
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

Carseats = pd.read_csv('../Carseats.csv')

Carseats['Urban']=Carseats['Urban'].map(dict({'Yes': 1, 'No': 0}))
Carseats['US']=Carseats['US'].map(dict({'Yes': 1, 'No': 0}))
#Carseats = pd.get_dummies(Carseats, columns=['Urban','US'], drop_first=True)

print(Carseats.columns)
print(Carseats)
y=Carseats['Sales']
print(y)
X=MS(['Price','Urban','US']).fit_transform(Carseats)
print(X)

model=sm.OLS(y,X)
results=model.fit()
print(summarize(results))

X=MS(['Price','US']).fit_transform(Carseats)
model1=sm.OLS(y,X)
results1=model1.fit()
print(summarize(results1))
print(anova_lm(results1,results))
print(results1.conf_int(alpha=0.05))

ax = subplots(figsize=(8,8))[1]
ax.scatter(results1.fittedvalues, results1.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');
plt.savefig(r"Outlier_careseats.pdf")
##predictions=model.predict(X)                                                                                                                                                                      
##table=anova_lm(model.fit())                                                                                                                                                                       
infl = results1.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)
plt.savefig(r"Leverage_careseats.pdf")
