# Import necessary libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import snap as sp
import numpy as np


file1 = pd.read_csv("/TAS-Com/results/communityConnectivityComparison.csv")

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(12, 6))
# plt.figure(figsize=(8, 8))

# #Draw the stacked bar
# algorithms = file1['Approach']
# m1 = file1['Q']
# m2 = file1['NMI']
#
# # Set the position of the bars on the x-axis
# bar_width = 0.6
# r = np.arange(len(algorithms))
#
# # Create the bars for Metric 1
# plt.bar(r, m1, color='blue', edgecolor='black', width=bar_width, label=r'$Q$')
#
# # Stack the bars for Metric 2 on top of Metric 1
# # plt.bar(r, m2, bottom=m1, color='orange', edgecolor='black', width=bar_width, label='Metric 2')


# sns.relplot(x="Epoch", y="Loss", kind="line", data=file1, hue="Network")
# sns.relplot(x="Generation", y="Fitness (Modularity)", kind="line", data=file1, legend=True)
# custom_palette = ['#00AFB9', 'orange']
# custom_palette = ['#00008B', '#984EA3']
# custom_palette = ['#1F77B4', '#FAD7AC']
custom_palette = ['#1F77B4', '#FFAE49']

# # custom_palette = ['#ADD8E6', '#87CEEB', '#1E90FF', '#4169E1', '#00008B'] - priginal
# custom_palette = ['#ADD8E6', '#739fc2', '#1E90FF', '#4169E1', '#00008B']
sns.barplot(x="Network", y="Overall_connectivity", hue="Approach", data=file1, palette=custom_palette, width=0.9)

# sns.barplot(x="Metric", y="Score", hue="Variant", data=file1, palette=custom_palette, width=0.7)



# sns.relplot(x="Generation", y="Fitness(Modularity)", kind="line", data=file1)
# sns.relplot(x="Generation", y="Fitness(Modularity)", kind="line", data=file1)
#sns.relplot(x="generation", y="maxfitness", kind="line", data=file1, color="red")
#sns.lineplot(x="Run", y="maxfitness", data=file2, color="red")
# axins = inset_axes(ax, width="40%", height="30%", loc='upper right')
# sns.relplot(data=file1, x="Generation", y="Fitness (Modularity)", kind="line", hue="Algorithm", legend=False, ax=axins)
# axins.set_xticklabels('')
# axins.set_yticklabels('')
# ax.set_xlabel(x_column)
# ax.set_ylabel(y_column)


# # plt.xlim(185, 200)
# # plt.ylim(0, 119.99)
plt.ylim(1, 52)
plt.yticks(np.arange(1, 52, 10))
# # plt.ylim(0.64478, 0.64488)
# # plt.xlabel('')
# # plt.ylabel('')


# plt.figure(figsize=(20, 10))
# plt.title('Convergence Curves')

plt.ylabel(r'$O_c(CS)$')
plt.tight_layout()
# plt.xaxis.grid(True, linestyle='--')
plt.grid(axis='y')
# plt.xticks(rotation=45)
plt.legend(loc="upper left")
# legend = plt.legend(loc='upper right', ncol=2)


plt.savefig("//local/scratch/PycharmProjects/DGCluster_OurMethod/plots/Community_Connectivity_starting1_test.pdf", format='pdf')

plt.show()


