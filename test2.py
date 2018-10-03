import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import pandas as pd

tips = sns.load_dataset("tips")

data_s = pd.read_csv('miller.csv')
print(data_s.corr())
