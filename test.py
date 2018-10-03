import seaborn as sns, matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
print(tips)

# sns.boxplot(x="day", y="total_bill", data=tips, palette="PRGn")
#
# # statistical annotation
# x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
# y, h, col = tips['total_bill'].max() + 10, 2, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "p < 0.01", ha='center', va='bottom', color=col)
#
# x1, x2 = 0, 3   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
# y, h, col = tips['total_bill'].max() + 2, 2, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "p < 0.05", ha='center', va='bottom', color=col)
#
# plt.show()