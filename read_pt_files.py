import torch

# Load the .pt file
file_path = '/TAS-Com/results/Leiden_30runs_lossL1/cora/results_cora_0.5_300_gcn_9.pt'
data = torch.load(file_path)
print(data)

