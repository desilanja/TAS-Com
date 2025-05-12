import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
file1 = pd.read_csv("/TAS-Com/results/Leiden_30runs_lossL1/cora/results_cora_0.8_epochsandlosses_DGCluster.csv")


# sns.relplot(x="Epoch", y="Loss", kind="line", data=file1, hue="Metric", legend=True)
sns.relplot(x="Epoch", y="Loss", kind="line", data=file1, legend=True)
plt.title('Cora (Lambda = 0.8)')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig("/TAS-Com/plots/convergence_curve_cora_0.8_DGCluster.pdf", format='pdf')
plt.show()