# TAS-Com: Topological and Attributive Similarity-based Community detection
# This code is releated to the paper titled: Community Detection based on Topological and Attributive Similarity using Graph Convolutional Neural Networks

## Requirements
- First install torch from https://pytorch.org/get-started/locally/
- Install networkx == 3.2.1
- Then run `install.py` using the following command to install all the required packages to execute TAS-Com

`python install.py`

## Reproducing results by TAS-Com

To generate the results for a particular dataset (e.g., cora) with mu parameter (e.g., 0.5) and a specific seed (e.g., 1) run the following command in terminal:

`python main.py --dataset cora --mu 0.5 --seed 1`

- Names of the datasets to be passed via arguments are: `cora`, `citeseer`, `photo`, `computers`, `coauthorcs`, `coauthorphysics`
- Default seed value is set to 0 and the experiments are carried out on 10 seeds i.e, {0, 1, 2, ..., 9}. 
- Reported results are related to the following settings of the mu parameter for each benchmark network.
| Network      | mu parameter setting |
|--------------|----------------------|
| Cora         |         0.5          |
| Citeseer     |         0.2          |
| Amazon Photo |         0.2          |
| Amazon PC    |         0.5          |
| Coauthor CS  |         10           |
| Coauthor Phy |         0.5          |


