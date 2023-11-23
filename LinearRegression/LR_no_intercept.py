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

rng = np.random.default_rng(1)
x = rng.normal(size=100)
y = 2 * x + rng.normal(size=100)
model=sm.OLS(y,x,intercept=False)
results=model.fit()
print(summarize(results))
print(results.conf_int(alpha=0.05))


model1=sm.OLS(x,y,intercept=False)
results1=model1.fit()
print(summarize(results1))
print(results1.conf_int(alpha=0.05))
