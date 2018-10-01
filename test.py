import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data1 = pd.read_excel('data.xlsx')
data1 = data1.dropna()
data1 = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
data1 = data1.dropna()

# fig, ax = plt.subplots(figsize=(10, 6))
sns.clustermap(data1.corr(), metric="correlation", method="single", cmap="vlag", linewidths=0.5)
plt.show()