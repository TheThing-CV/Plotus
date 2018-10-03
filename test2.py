import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

tips = sns.load_dataset("tips")

a_string = 'total_bill+tip~sex'
print(a_string.split('~')[0].split('+'))
