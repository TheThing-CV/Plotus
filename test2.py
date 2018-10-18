import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit, probit, poisson, ols
import statsmodels.formula.api as smf
import seaborn as sns
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

data = sns.load_dataset('tips')
data = data.dropna()
#print(data.head())

print(data['sex'].unique())

data['Male'] = data['sex'].map({data['sex'].unique()[0]: 1, data['sex'].unique()[1]: 0})
# cleanup_nums = {"sex":     {"Female": 0, "Male": 1}}
# data.replace(cleanup_nums, inplace=True)



# print(data.head())
#
model = logit("Male ~ tip", data).fit()
#print(dir(model))
print(model.bse)
print(model.summary())


# binary = pd.get_dummies(data['sex'])
# print(binary)
# print(sm.datasets.fair.SOURCE)
# dta = sm.datasets.fair.load_pandas().data
# print(dta.head())
# dta['affair'] = (dta['affairs'] > 0).astype(float)
#
# affair_mod = logit("affair ~ occupation + educ + occupation_husb", dta).fit()
# print(affair_mod.summary())

#
# tips = sns.load_dataset("tips")
# tips = tips.dropna()
# print(tips.head())
# model = sm.Logit(tips['sex'], tips['tip'])
#
# result = model.fit()
#
# print(result.summary2())
