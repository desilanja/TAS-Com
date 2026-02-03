import csv
from collections import defaultdict

import igraph
import numpy as np
import random

import pandas
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
from community import community_louvain
from networkx import connected_components
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor


from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv

import scipy.sparse
from sklearn.cluster import Birch, SpectralClustering, KMeans

import networkx as nx

import argparse
import utils
import os
import leidenalg as la
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
from cdlib import algorithms as cdalg
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from itertools import combinations


def parse_args():
    args = argparse.ArgumentParser(description='TAS-Com arguments.')
    args.add_argument('--dataset', type=str, default='cora')
    args.add_argument('--mu', type=float, default=0.5)
    args.add_argument('--alp', type=float, default=0.0)
    args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    args.add_argument('--epochs', type=int, default=300)
    args.add_argument('--base_model', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
    args.add_argument('--seed', type=int, default=0)
    args = args.parse_args()
    return args


def load_dataset(dataset_name):
    if dataset_name == 'cora':
        dataset = Planetoid(root='data', name="Cora")
    elif dataset_name == 'citeseer':
        dataset = Planetoid(root='data', name="Citeseer")
    elif dataset_name == 'computers':
        dataset = Amazon(root='data', name='Computers')
    elif dataset_name == 'photo':
        dataset = Amazon(root='data', name='Photo')
    elif dataset_name == 'coauthorcs':
        dataset = Coauthor(root='data/Coauthor', name='CS')
    elif dataset_name == 'coauthorphysics':
        dataset = Coauthor(root='data/Coauthor', name='Physics')
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} not implemented.')
    return dataset


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, base_model):
        super(GNN, self).__init__()

        if base_model == 'gcn':
            self.conv1 = GCNConv(in_dim, 256)
            self.conv2 = GCNConv(256, 128)
            self.conv3 = GCNConv(128, out_dim)
        elif base_model == 'gat':
            self.conv1 = GATConv(in_dim, 256)
            self.conv2 = GATConv(256, 128)
            self.conv3 = GATConv(128, out_dim)
        elif base_model == 'gin': # Graph Isormorphism Network
            self.conv1 = GINConv(nn.Linear(in_dim, 256))
            self.conv2 = GINConv(nn.Linear(256, 128))
            self.conv3 = GINConv(nn.Linear(128, out_dim))
        elif base_model == 'sage': # Graph sample and aggregation
            self.conv1 = SAGEConv(in_dim, 256)
            self.conv2 = SAGEConv(256, 128)
            self.conv3 = SAGEConv(128, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        # Transformation of the embeddings to avoid the negative values
        x = x / (x.sum())  # To prevent vanishing gradients because of the tanh activation function for large values.
        x = (F.tanh(x)) ** 2 # Ensures the output is constrained within the positive coordinate space
        # x = (F.relu(x)) ** 2
        x = F.normalize(x)  # L2 normalization - reduces the cosine similarity computation of node pairs to corresponding dot products

        return x


def convert_scipy_torch_sp(sp_adj):
    sp_adj = sp_adj.tocoo()
    indices = torch.tensor(np.vstack((sp_adj.row, sp_adj.col)))
    sp_adj = torch.sparse_coo_tensor(indices, torch.tensor(sp_adj.data), size=sp_adj.shape)
    return sp_adj

# Leiden_based objective
def leiden_objective(output, s):
    sample_size = len(s)

    out = output[s, :].float()

    # C = oh_labels[s, :].float()
    C = oh_com_leiden_labels[s, :].float()

    X = C.sum(dim=0)
    X = X ** 2
    X = X.sum()

    Y = torch.matmul(torch.t(out), C)
    Y = torch.matmul(Y, torch.t(Y))
    Y = torch.trace(Y)

    t1 = torch.matmul(torch.t(C), C)  # H = C.transpose(C)
    t1 = torch.matmul(t1, t1)
    t1 = torch.trace(t1)

    t2 = torch.matmul(torch.t(out), out)  # Xs.transpose(Xs)
    t2 = torch.matmul(t2, t2)
    t2 = torch.trace(t2)

    t3 = torch.matmul(torch.t(out), C)
    t3 = torch.matmul(t3, torch.t(t3))
    t3 = torch.trace(t3)

    leiden_objective_loss = 1 / (sample_size ** 2) * (t1 + t2 - 2 * t3)

    return leiden_objective_loss

def aux_objective(output, s):
    sample_size = len(s)

    out = output[s, :].float()

    C = oh_labels[s, :].float() # Ground truth labels - original method
    # C = oh_com_leiden_labels[s, :].float() # Communities obtained by Leiden

    X = C.sum(dim=0)
    X = X ** 2
    X = X.sum()

    Y = torch.matmul(torch.t(out), C)
    Y = torch.matmul(Y, torch.t(Y))
    Y = torch.trace(Y)

    t1 = torch.matmul(torch.t(C), C) # H = C.transpose(C)
    t1 = torch.matmul(t1, t1)
    t1 = torch.trace(t1)

    t2 = torch.matmul(torch.t(out), out) # Xs.transpose(Xs)
    t2 = torch.matmul(t2, t2)
    t2 = torch.trace(t2)

    t3 = torch.matmul(torch.t(out), C)
    t3 = torch.matmul(t3, torch.t(t3))
    t3 = torch.trace(t3)

    aux_objective_loss = 1 / (sample_size ** 2) * (t1 + t2 - 2 * t3)

    return aux_objective_loss


def regularization(output, s):
    out = output[s, :]
    ss = out.sum(dim=0)
    ss = ss ** 2
    ss = ss.sum()
    avg_sim = 1 / (len(s) ** 2) * ss

    return avg_sim ** 2

# new L1 loss function based on the Leiden output
def loss_fn(output, lam=0.0, alp=0.0, epoch=-1):
    record = []
    sample_size = int(1 * num_nodes)
    s = random.sample(range(0, num_nodes), sample_size)

    # s_output = output[s, :]
    #
    # s_adj = sparse_adj[s, :][:, s]
    # s_adj = convert_scipy_torch_sp(s_adj)
    # s_degree = degree[s]
    #
    # x = torch.matmul(torch.t(s_output).double(), s_adj.double().to(device))
    # x = torch.matmul(x, s_output.double())
    # x = torch.trace(x)
    #
    # y = torch.matmul(torch.t(s_output).double(), s_degree.double().to(device))
    # y = (y ** 2).sum()
    # y = y / (2 * num_edges)
    #
    # # scaling=1
    # scaling = num_nodes ** 2 / (sample_size ** 2)
    #
    # m_loss = -((x - y) / (2 * num_edges)) * scaling # Modularity loss based loss function

    m_loss = leiden_objective(output, s)
    # m_loss = 0
    # lam = 0
    aux_objective_loss = aux_objective(output, s)

    aux_loss = lam * aux_objective_loss # Attribute similarity based loss function

    reg_loss = alp * regularization(output, s) # regularization term does not make an impact. Therefore it is considered as zero.

    loss = m_loss + aux_loss + reg_loss

    print('epoch: ', epoch, 'loss: ', loss.item(), 'm_loss: ', m_loss.item(), 'aux_loss: ', aux_loss.item(), 'reg_loss: ', reg_loss.item())
    # print('epoch: ', epoch, 'loss: ', loss.item(), 'm_loss: ', '0', 'aux_loss: ', aux_loss.item(),
    #       'reg_loss: ', reg_loss.item())

    return loss, m_loss, aux_objective_loss, aux_loss

# DGCluter original loss function
# def loss_fn(output, lam=0.0, mu=1.0, alp=0.0, epoch=-1):
# def loss_fn(output, lam=0.0, alp=0.0, epoch=-1, run=-1):
#     record = []
#     sample_size = int(1 * num_nodes)
#     s = random.sample(range(0, num_nodes), sample_size)
#
#     s_output = output[s, :]
#
#     s_adj = sparse_adj[s, :][:, s]
#     s_adj = convert_scipy_torch_sp(s_adj)
#     s_degree = degree[s]
#
#     x = torch.matmul(torch.t(s_output).double(), s_adj.double().to(device))
#     x = torch.matmul(x, s_output.double())
#     x = torch.trace(x)
#
#     y = torch.matmul(torch.t(s_output).double(), s_degree.double().to(device))
#     y = (y ** 2).sum()
#     y = y / (2 * num_edges)
#
#     # scaling=1
#     scaling = num_nodes ** 2 / (sample_size ** 2)
#
#     m_loss = -((x - y) / (2 * num_edges)) * scaling # Modularity loss based loss function
#     aux_objective_loss = aux_objective(output, s)
#     aux_loss = lam * aux_objective_loss # Attribute similarity based loss function
#
#
#     reg_loss = alp * regularization(output, s) # regularization term does not make an impact. Therefore it is considered as zero.
#
#     # loss = mu * m_loss + aux_loss + reg_loss
#     loss = m_loss + aux_loss + reg_loss
#     # record.append([run, epoch, loss.item()])
#     # csv_file = f'results/Leiden_30runs_lossL1/{dataset_name}/results_{dataset_name}_{lam}_epochsandlosses_DGCluster.csv'
#     # with open(csv_file, mode='a', newline='') as file:
#     #     writer = csv.writer(file)
#     #     # writer.writerow(['Run', 'Epoch', 'Loss'])
#     #     writer.writerows(record)
#
#     print('epoch: ', epoch, 'loss: ', loss.item(), 'm_loss: ', m_loss.item(), 'aux_loss: ', aux_loss.item(), 'reg_loss: ', reg_loss.item())
#
#     return loss, m_loss, aux_objective_loss, aux_loss

# Split the network for training and testing - Randomly split the nodes
def split_network_randomly(G, train_ratio=0.8):
    # Get all the nodes in the network
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    # Determine the number of nodes in the training set
    train_size = int(train_ratio * num_nodes)

    # Shuffle the nodes and split into training and testing sets
    random.shuffle(nodes)
    train_nodes = nodes[:train_size]
    test_nodes = nodes[train_size:]

    # Create subgraphs for training and testing
    G_train = G.subgraph(train_nodes).copy()
    G_test = G.subgraph(test_nodes).copy()

    return G_train, G_test

def membership_to_communities(membership_list):
    communities = {}

    for node, community in enumerate(membership_list):
        if community not in communities:
            communities[community] = []
        communities[community].append(node)

    return communities


def random_train_test_split(nodes, train_size):
    if len(nodes) == 0:
        return [], nodes
    elif len(nodes) == 1:
        return nodes, []
    else:
        # Ensure at least one node in the training set
        train_size_adjusted = max(1 / len(nodes), train_size)
        train_nodes, test_nodes = train_test_split(nodes, train_size=train_size_adjusted, random_state=42)
        return train_nodes, test_nodes
    # train_nodes, test_nodes = train_test_split(nodes, train_size=train_size, random_state=42)
    return train_nodes, test_nodes

# Split the network for training and testing using the spectral clustering
def split_network_SC(adj, n_clusters = 2, train_size = 0.8):
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    cluster_labels = sc.fit_predict(adj)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for cluster_id in range(n_clusters):
        cluster_nodes = np.where(cluster_labels == cluster_id)[0]
        train_nodes, test_nodes = random_train_test_split(cluster_nodes, train_size)
        train_mask[train_nodes] = True
        test_mask[test_nodes] = True
    return train_mask, test_mask


def quantitative_metrics(train_embeddings, test_embeddings):
    cosine_sim = cosine_similarity(train_embeddings, test_embeddings)
    euclidean_dist = euclidean_distances(train_embeddings, test_embeddings)

    print(f'Average Cosine Similarity: {cosine_sim.mean()}')
    print(f'Average Euclidean Distance: {euclidean_dist.mean()}')


def compute_centroids(features, labels):
    # features = normalize(features, norm='l1', axis=1)
    # tfidf_transformer = TfidfTransformer()
    # features = tfidf_transformer.fit_transform(features).toarray()
    unique_labels = np.unique(labels)
    centroids = np.zeros((len(unique_labels), features.shape[1]))
    for i, label in enumerate(unique_labels):
        centroids[i] = np.mean(features[labels == label], axis=0)
    return centroids

# def merge_communities(features, labels, similarity_threshold=0.9):
#     centroids = compute_centroids(features, labels)
#     similarity_matrix = cosine_similarity(centroids)
#     merged_labels = labels.copy()
#     merge_map = {}
#
#     for i in range(similarity_matrix.shape[0]):
#         for j in range(i + 1, similarity_matrix.shape[1]):
#             if similarity_matrix[i, j] >= similarity_threshold:
#                 # Merge community j into community i
#                 merge_map[j] = i
#
#     for old_label, new_label in merge_map.items():
#         merged_labels[labels == old_label] = new_label
#
#     # Reassign labels to ensure they are continuous
#     unique_labels = np.unique(merged_labels)
#     new_labels = np.zeros_like(merged_labels)
#     for new_label, unique_label in enumerate(unique_labels):
#         new_labels[merged_labels == unique_label] = new_label
#
#     return new_labels

def community_to_membership(individual):
    new_com_dict = {}
    for k in range(0, len(individual)):
        for element in individual[k]:
            new_com_dict[int(element)] = k
        # print("new community dictionary: ", new_com_dict)
    sort_com_dict = sorted(new_com_dict)
    part_membership = [new_com_dict[key] for key in sort_com_dict]
    return part_membership


def merge_leiden_communities(G, partition, GTCommunities):
    best_modularity = nx.algorithms.community.modularity(G, GTCommunities)
    best_partition = GTCommunities
    best_leiden_subgraph = partition

    for com1, com2 in combinations(partition, 2):
        new_partition = [com for com in GTCommunities if com != com1 and com != com2]
        new_leiden_subgraph = [com for com in partition if com != com1 and com != com2]
        merged_community = com1 + com2
        new_leiden_subgraph.append(merged_community)
        new_partition.append(merged_community)


        modularity = nx.algorithms.community.modularity(G, new_partition)
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = new_partition
            best_leiden_subgraph = new_leiden_subgraph

    return best_partition, best_leiden_subgraph

def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
# def merge_leiden_communities_basedon_localQ(G, partition):
#     subgraph_leiden_partition = G.subgraph(flatten_list_of_lists(partition))
#     best_modularity = nx.algorithms.community.modularity(subgraph_leiden_partition, partition)
#     best_partition = partition
#     # best_leiden_subgraph = partition
#
#     for com1, com2 in combinations(partition, 2):
#         new_partition = [com for com in partition if com != com1 and com != com2]
#         # new_leiden_subgraph = [com for com in partition if com != com1 and com != com2]
#         merged_community = com1 + com2
#         # new_leiden_subgraph.append(merged_community)
#         new_partition.append(merged_community)
#
#
#         modularity = nx.algorithms.community.modularity(subgraph_leiden_partition, new_partition)
#         if modularity > best_modularity:
#             best_modularity = modularity
#             best_partition = new_partition
#             # best_leiden_subgraph = new_leiden_subgraph
#
#     return best_partition, best_modularity

# def merge_smallest_communities(G, partition):
#     subgraph_leiden_partition = G.subgraph(flatten_list_of_lists(partition))
#     best_modularity = nx.algorithms.community.modularity(subgraph_leiden_partition, partition)
#     best_partition = partition
#
#     while len(best_partition) > 10:
#         # Find the two smallest communities to merge
#         community_sizes = [(len(best_partition[i]), i) for i in range(len(best_partition))]
#         community_sizes.sort()
#         smallest_community_index = community_sizes[0][1]
#         next_smallest_community_index = community_sizes[1][1]
#
#         # Merge the two smallest communities
#         new_partition = partition[:smallest_community_index] + partition[smallest_community_index+1:]
#         new_partition[next_smallest_community_index] += partition[smallest_community_index]
#
#         # Calculate the new modularity
#         new_modularity = nx.algorithms.community.modularity(subgraph_leiden_partition, new_partition)
#         if new_modularity > best_modularity:
#             best_modularity = new_modularity
#             best_partition = new_partition
#
#     return best_partition, best_modularity

def merge_leiden_communities_max_modularity(subG, partition):
    # original_partition = partition
    components = list(connected_components(subG))
    # threshold_com_size = int(len(partition)/4)
    threshold_com_size = (len(components)/2)
    # threshold_com_size = int(len(partition)/4)
    # threshold_com_size = 3
    if len(partition) > 10:
        while len(partition) > threshold_com_size:
            community_combinations = {}
            key = 0
            for com1, com2 in combinations(partition, 2):
                new_partition = [com for com in partition if com != com1 and com != com2]
                # new_leiden_subgraph = [com for com in partition if com != com1 and com != com2]
                merged_community = com1 + com2
                # new_leiden_subgraph.append(merged_community)
                new_partition.append(merged_community)
                Q = nx.algorithms.community.modularity(subG, new_partition)
                community_combinations[key] = {'communityStructure': new_partition, 'modularity': Q}
                key += 1
            sorted_community_combinations = sorted(community_combinations.items(), key=lambda item: item[1]['modularity'], reverse=True)
            # Get the community structure with the highest modularity
            best_iteration, best_result = sorted_community_combinations[0]
            partition = best_result['communityStructure']
    return partition

def obtain_Leiden_communities(Graph):
    lei = 0
    Q_leiden_max = -1
    for lei in range(10):
        com_leiden_label = la.find_partition(igraph.Graph.from_networkx(Graph),
                                             la.ModularityVertexPartition,
                                             )
        com_leiden_label = np.array(com_leiden_label.membership)
        # NMI_leiden = utils.compute_nmi(com_leiden_label, data.y.squeeze().cpu().numpy())
        Q_leiden = utils.compute_fast_modularity(com_leiden_label, num_nodes, num_edges, torch_sparse_adj, degree,
                                                 device)
        if Q_leiden >= Q_leiden_max:
            Q_leiden_max = Q_leiden
            sel_com_leiden_label = com_leiden_label
    leiden_communities = membership_to_communities(sel_com_leiden_label)
    return list(leiden_communities.values())


def preprocess_groundtruth_Leiden(G, GTcommunities):
    # Sort the dictionary by keys
    sorted_GTcommunities = dict(sorted(GTcommunities.items()))
    leiden_communities = []
    leiden_GTcommunities = {}
    for key in sorted_GTcommunities:
        com_subgraph = G.subgraph(list(sorted_GTcommunities[key]))
        leiden_subgraph_com = cdalg.leiden(com_subgraph).communities
        if key == 0:
            new_GTCommunities = list(sorted_GTcommunities.values())
        com_to_remove = list(sorted_GTcommunities[key])
        # new_GTCommunities = new_GTCommunities.remove(com_to_remove)
        new_GTCommunities = [lst for lst in new_GTCommunities if lst != com_to_remove]
        for com in leiden_subgraph_com:
            new_GTCommunities.append(com)
        length_leiden_subgraph_com = len(leiden_subgraph_com)
        if len(leiden_subgraph_com) > 10:
            count = 0
            c = 0
            previous_no_com = length_leiden_subgraph_com
            while len(leiden_subgraph_com) > 40:
                new_GTCommunities, leiden_subgraph_com = merge_leiden_communities(G, leiden_subgraph_com, new_GTCommunities)
                if len(leiden_subgraph_com) != previous_no_com:
                    previous_no_com = len(leiden_subgraph_com)
                else:
                    c += 1
                if c == 5:
                    break
                print(f'count: {count}')
                count += 1
    new_GTCommunities = community_to_membership(new_GTCommunities)
    return torch.tensor(new_GTCommunities)

# def preprocess_groundtruth_Leiden_local_Q(G, GTcommunities): # GT communities are refined using Leiden and merge based on the local modularity optimization
#     # Sort the dictionary by keys
#     sorted_GTcommunities = dict(sorted(GTcommunities.items()))
#     leiden_communities = []
#     leiden_GTcommunities = {}
#     for key in sorted_GTcommunities:
#         com_subgraph = G.subgraph(list(sorted_GTcommunities[key]))
#         leiden_subgraph_com = cdalg.leiden(com_subgraph).communities
#         count = 0
#         while len(leiden_subgraph_com) > 10:
#             # leiden_subgraph_com, modularity = merge_leiden_communities_basedon_localQ(G, leiden_subgraph_com)
#             leiden_subgraph_com, modularity = merge_smallest_communities(G, leiden_subgraph_com)
#             print(f'Modularity {count}: {modularity}')
#             count += 1
#             print(f'count: {count}')
#         print('done')
#
#         if key == 0:
#             new_GTCommunities = list(sorted_GTcommunities.values())
#         com_to_remove = list(sorted_GTcommunities[key])
#         # new_GTCommunities = new_GTCommunities.remove(com_to_remove)
#         new_GTCommunities = [lst for lst in new_GTCommunities if lst != com_to_remove]
#         for com in leiden_subgraph_com:
#             new_GTCommunities.append(com)
#         length_leiden_subgraph_com = len(leiden_subgraph_com)
#         if len(leiden_subgraph_com) > 10:
#             count = 0
#             c = 0
#             previous_no_com = length_leiden_subgraph_com
#             while len(leiden_subgraph_com) > 40:
#                 new_GTCommunities, leiden_subgraph_com = merge_leiden_communities(G, leiden_subgraph_com, new_GTCommunities)
#                 if len(leiden_subgraph_com) != previous_no_com:
#                     previous_no_com = len(leiden_subgraph_com)
#                 else:
#                     c += 1
#                 if c == 5:
#                     break
#                 print(f'count: {count}')
#                 count += 1
#     new_GTCommunities = community_to_membership(new_GTCommunities)
#     return torch.tensor(new_GTCommunities)

def preprocess_groundtruth_Leiden_local_Q(G, GTcommunities): # GT communities are refined using Leiden and merge based on the local modularity optimization
    # Sort the dictionary by keys
    sorted_GTcommunities = dict(sorted(GTcommunities.items()))
    leiden_communities = []
    # leiden_GTcommunities = {}
    for key in sorted_GTcommunities:
        com_subgraph = G.subgraph(list(sorted_GTcommunities[key]))
        leiden_subgraph_com = cdalg.leiden(com_subgraph).communities
        # leiden_subgraph_com = obtain_Leiden_communities(com_subgraph)
        if len(leiden_subgraph_com) > 10:
            new_subCS = merge_leiden_communities_max_modularity(com_subgraph, leiden_subgraph_com)
            for c in new_subCS:
                leiden_communities.append(c)
        else:
            leiden_communities.append(list(sorted_GTcommunities[key]))
    print(f'Leiden GT communities: {len(leiden_communities)}')
    leiden_GTcommunities = community_to_membership(leiden_communities)
    return torch.tensor(leiden_GTcommunities)

def preprocess_groundtruth_connectedComponents(G, node_labels):
    community_components = {}
    unique_labels = set(node_labels)

    for label in unique_labels:
        # Get nodes in the current community
        community_nodes = [i for i, lbl in enumerate(node_labels) if lbl == label]

        # Create a subgraph for the community
        subgraph = G.subgraph(community_nodes)

        # Find connected components in the subgraph
        components = list(connected_components(subgraph))

        # Store the components
        community_components[label] = components

    # Print the disconnected components for each community
    new_communities = []
    for label, components in community_components.items():
        # print(f"Community {label}:")
        for component in components:
            # print(f"  Component of size {len(component)}: {component}")
            new_communities.append(list(component))
    new_communities = community_to_membership(new_communities)

    return torch.tensor(new_communities)

# def train(model, optimizer, data, epochs, lam, alp):
def train(model, optimizer, data, epochs, lam, alp):
    # recordL = []
    # recordL1 = []
    # recordL2 = []
    # recordlamL2 = []
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs) # used to adjust the learning rate during training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data) # Node embedding at a specific epoch

        loss, m_loss, aux_objective_loss, aux_loss = loss_fn(out, lam, alp, epoch) # Original loss function
        # loss = loss_fn(out, lam, mu, alp, epoch)
        loss.backward() # Computes the gradients of the loss with respect to the model parameters (Performs backpropagation)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # clip (limit) the norm of the gradients of a model's parameters
        optimizer.step() # Update the model's weights and biases
        scheduler.step() # change the learning rate
        cur_learning_rate = scheduler.get_last_lr() # new learning rate
        print('Learning rate: ', cur_learning_rate)
        # recordL.append([run, epoch, loss.item()])
        # recordL1.append([run, epoch, m_loss.item()])
        #
        # recordL2.append([run, epoch, aux_objective_loss.item()])
        # recordlamL2.append([run, epoch, aux_loss.item()])

    # # csv_file = f'results/Leiden_30runs_lossL1/{dataset_name}/Weighted_LeidenCluster_loss/DGCluster/results_{dataset_name}_{lam}_epochsandlosses.csv'
    # csv_file1 = f'results/processedGT_basedonLeiden/{dataset_name}/LossVsEpochs/results_{dataset_name}_{lam}_lossinepochs.csv'
    # with open(csv_file1, mode='a', newline='') as file1:
    #     writer = csv.writer(file1)
    #     # writer.writerow(['Run', 'Epoch', 'Loss'])
    #     writer.writerows(recordL)
    #
    # csv_file2 = f'results/processedGT_basedonLeiden/{dataset_name}/LossVsEpochs/results_{dataset_name}_{lam}_lossinepochs_L1.csv'
    # with open(csv_file2, mode='a', newline='') as file2:
    #     writer = csv.writer(file2)
    #     # writer.writerow(['Run', 'Epoch', 'Loss'])
    #     writer.writerows(recordL1)
    #
    # csv_file3 = f'results/processedGT_basedonLeiden/{dataset_name}/LossVsEpochs/results_{dataset_name}_{lam}_lossinepochs_L2.csv'
    # with open(csv_file3, mode='a', newline='') as file3:
    #     writer = csv.writer(file3)
    #     # writer.writerow(['Run', 'Epoch', 'Loss'])
    #     writer.writerows(recordL2)
    #
    # csv_file4 = f'results/processedGT_basedonLeiden/{dataset_name}/LossVsEpochs/results_{dataset_name}_{lam}_lossinepochs_lamL2.csv'
    # with open(csv_file4, mode='a', newline='') as file4:
    #     writer = csv.writer(file4)
    #     # writer.writerow(['Run', 'Epoch', 'Loss'])
    #     writer.writerows(recordlamL2)

    # clusters = Birch(n_clusters=None, threshold=0.5).fit_predict(out.detach().cpu().numpy(),
    #                                                              y=None)  # CLusters obtained by Birch cllustering algorithm.
    # # kmeans = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(out.detach().cpu().numpy())
    # # clusters = kmeans.labels_
    # FQ = utils.compute_fast_modularity(clusters, num_nodes, num_edges, torch_sparse_adj, degree,
    #                                    device)  # Compute modularity of the obtained community structure
    # print('------------Training Performance--------------------')
    # print('No of clusters: ', max(clusters) + 1)
    # print('Modularity:', FQ)
    #
    # NMI = utils.compute_nmi(clusters, data.y.squeeze().cpu().numpy())  # Compute NMI
    # print('NMI:', NMI)
    #
    # conductance = utils.compute_conductance(clusters, Graph)  # Compute conductance
    # avg_conductance = sum(conductance) / len(conductance)
    # print('Conductance: ', avg_conductance * 100)
    #
    # f1_score = utils.sample_f1_score(data, clusters, num_nodes)  # Compute f1_score
    # print('Sample_F1_score:', f1_score)
    # print('------------------------------------------')
    # results = [run, FQ, NMI, avg_conductance, f1_score, np.unique(clusters).shape[0]]
    # csv_file2 = f'results/Leiden_Com_Merge_Similaritybased/{dataset_name}/similarity_threshold_0.7/results_{dataset_name}_{lam}_TrainingPerformance.csv'
    # with open(csv_file2, mode='a', newline='') as file2:
    #     writer = csv.writer(file2)
    #     # writer.writerow(['Run', 'Modularity', 'NMI', 'Conductance', 'F1_score', 'num_clusters'])
    #     writer.writerow(results)
    return out




if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    lam = args.mu
    alp = args.alp
    epochs = args.epochs
    device = args.device
    base_model = args.base_model
    seed = args.seed


    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # device selection
    if torch.cuda.is_available() and device != 'cpu':
        device = torch.device(device)
    else:
        device = torch.device('cpu')
        print(f'Using device: {device}')

    # transform data
    transform = T.NormalizeFeatures()

    # load dataset
    dataset = load_dataset(dataset_name)
    data = dataset[0]
    data = data.to(device)

    # preprocessing
    num_nodes = data.x.shape[0]
    num_edges = (data.edge_index.shape[1])



    sparse_adj = sp.sparse.csr_matrix((np.ones(num_edges), data.edge_index.cpu().numpy()), shape=(num_nodes, num_nodes))
    torch_sparse_adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(num_edges).to(device), size=(num_nodes, num_nodes))
    degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float().to(device)
    Graph = nx.from_scipy_sparse_array(sparse_adj, create_using=nx.Graph).to_undirected()
    num_edges = int((data.edge_index.shape[1]) / 2)

    # labels = data.y.flatten()
    # # labels = preprocess_groundtruth_Leiden(Graph, labels)
    # num_classes = max(labels) + 1
    # oh_labels = F.one_hot(labels, num_classes=num_classes)  # Represents as a matrix (n * k)

    labels_original = data.y.flatten()
    labels_modified = preprocess_groundtruth_Leiden(Graph, membership_to_communities(labels_original.tolist()))
    # labels_modified = preprocess_groundtruth_Leiden_local_Q(Graph, membership_to_communities(labels_original.tolist()))
    # labels_modified = preprocess_groundtruth_connectedComponents(Graph, labels_original.tolist())
    oh_labels = F.one_hot(labels_modified, num_classes=max(labels_modified) + 1)
    # oh_labels = F.one_hot(labels_original, num_classes=max(labels_original) + 1)

    # ground_truth_Q = utils.compute_fast_modularity(labels, num_nodes, num_edges, torch_sparse_adj, degree, device)





    # # Obtain the community structure by Leiden - Maximizing NMI
    lei = 0
    NMI_leiden_max = 0
    for lei in range(30):
        com_leiden_label = la.find_partition(igraph.Graph.from_networkx(Graph), la.ModularityVertexPartition, initial_membership=labels_original).membership
        NMI_leiden = utils.compute_nmi(com_leiden_label, data.y.squeeze().cpu().numpy())
        # Q_leiden = utils.compute_fast_modularity(com_leiden_label, num_nodes, num_edges, torch_sparse_adj,
        #                                          degree, device)
        if NMI_leiden >= NMI_leiden_max:
            NMI_leiden_max = NMI_leiden
            sel_com_leiden_label = com_leiden_label

    # # Perform Leiden with a controlled steps for the optimization
    # lei = 0
    # NMI_leiden_max = 0
    # for lei in range(30):
    #     # partition = la.ModularityVertexPartition(igraph.Graph.from_networkx(Graph))
    #     # optimiser = la.Optimiser()
    #     com_leiden_label = la.find_partition(igraph.Graph.from_networkx(Graph),
    #                                          la.ModularityVertexPartition,
    #                                          initial_membership=labels_original)
    #     com_leiden_label = np.array(com_leiden_label.membership)
    #     NMI_leiden = utils.compute_nmi(com_leiden_label, data.y.squeeze().cpu().numpy())
    #     # Q_leiden = utils.compute_fast_modularity(com_leiden_label, num_nodes, num_edges, torch_sparse_adj,
    #     #                                          degree, device)
    #     # Q_leiden_nx = nx.algorithms.community.modularity(Graph, list(membership_to_communities(com_leiden_label).values()))
    #     # Q_GT = utils.compute_fast_modularity(labels, num_nodes, num_edges, torch_sparse_adj,
    #     #                                          degree, device)
    #     # Q_GT_nx = nx.algorithms.community.modularity(Graph, membership_to_communities(labels.numpy()))
    #     if NMI_leiden >= NMI_leiden_max:
    #         NMI_leiden_max = NMI_leiden
    #         sel_com_leiden_label = com_leiden_label

    # Obtain the community structure by Leiden - Maximizing Modularity
    # lei = 0
    # Q_leiden_max = 0
    # for lei in range(30):
    #     com_leiden_label = la.find_partition(igraph.Graph.from_networkx(Graph),
    #                                          la.ModularityVertexPartition,
    #                                          )
    #     com_leiden_label = np.array(com_leiden_label.membership)
    #     # NMI_leiden = utils.compute_nmi(com_leiden_label, data.y.squeeze().cpu().numpy())
    #     Q_leiden = utils.compute_fast_modularity(com_leiden_label, num_nodes, num_edges, torch_sparse_adj, degree, device)
    #     if Q_leiden >= Q_leiden_max:
    #         Q_leiden_max = Q_leiden
    #         sel_com_leiden_label = com_leiden_label

    # Obtain the community structure by Louvain - Maximizing NMI
    # lei = 0
    # NMI_leiden_max = 0
    # for lei in range(30):
    #     # com_leiden_label = la.find_partition(igraph.Graph.from_networkx(Graph),
    #     #                                      la.ModularityVertexPartition,
    #     #                                      initial_membership=labels).membership
    #     partition = community_louvain.best_partition(Graph)
    #     com_leiden_label = [partition[i] for i in range(len(Graph))]
    #     NMI_leiden = utils.compute_nmi(com_leiden_label, data.y.squeeze().cpu().numpy())
    #     # Q_leiden = utils.compute_fast_modularity(com_leiden_label, num_nodes, num_edges, torch_sparse_adj,
    #     #                                          degree, device)
    #     if NMI_leiden >= NMI_leiden_max:
    #         NMI_leiden_max = NMI_leiden
    #         sel_com_leiden_label = com_leiden_label


    # old_num_classes = max(sel_com_leiden_label) + 1
    # new_merged_Leiden_label = merge_communities(data.x.numpy(), sel_com_leiden_label,
    #                                             similarity_threshold=0.7)
    # new_num_classes = max(new_merged_Leiden_label) + 1

    new_merged_Leiden_label = torch.tensor(sel_com_leiden_label)
    oh_com_leiden_labels = F.one_hot(new_merged_Leiden_label, num_classes=max(new_merged_Leiden_label)+1)

    in_dim = data.x.shape[1]
    out_dim = 64
    model = GNN(in_dim, out_dim, base_model=base_model).to(device)

    optimizer_name = "Adam"
    lr = 1e-3
    param = list(model.parameters())
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001, amsgrad=True)

    # train(model, optimizer, data, epochs, lam, alp) # Original train()
    trained_embedding = train(model, optimizer, data, epochs, lam, alp)
    # df_t = pandas.DataFrame(trained_embedding.detach().numpy())
    # df_t.to_csv('results/Leiden_30runs_lossL1/TestingTrainingPerformance_default splitting/OurMethod/' + f'trainedEmbeddedMatrix_{dataset_name}_{lam}_OurMethod.csv', index=False)

    test_data = data.clone() # CLone the data set to be the test_data
    print(test_data)

    model.eval()
    x = model(test_data) # Give the same dataset to the model and obtain the node embeddings (e.g. 2708, 64)
    # loss_test = loss_fn(x, lam, alp, run)
    # df = pandas.DataFrame(x.detach().numpy())
    # df.to_csv(
    #     'results/Leiden_30runs_lossL1/TestingTrainingPerformance_default splitting/OurMethod/' + f'testEmbeddedMatrix_{dataset_name}_{lam}_OurMethod.csv',
    #     index=False)

    # print("Loss for test: ", loss_test)

    # quantitative_metrics(trained_embedding.detach().numpy(), x.detach().numpy())


    clusters = Birch(n_clusters=None, threshold=0.5).fit_predict(x.detach().cpu().numpy(), y=None) # CLusters obtained by Birch cllustering algorithm.
    # kmeans = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(x.detach().cpu().numpy())
    # clusters = kmeans.labels_

    cluster_dict = defaultdict(list)
    for node, cluster in enumerate(clusters):
        cluster_dict[cluster].append(node)

    # Check connectivity for each cluster
    connected_clusters = {}
    n_clusters = 0

    total_n_components = 0
    connected_count = 0
    con_components = []
    for cluster, nodes in cluster_dict.items():
        subgraph = Graph.subgraph(nodes)
        # Find connected components in the subgraph
        components = list(connected_components(subgraph))
        n_components = len(components)
        total_n_components = total_n_components + n_components
        n_clusters += 1
        for c in components:
            con_components.append(c)
        # n_clusters = n_clusters + n_components
        # if nx.is_connected(subgraph):
        #     connected_count += 1
        connected_clusters[cluster] = nx.is_connected(subgraph)
    # con_component_ratio = connected_count / n_clusters
    con_component_ratio = total_n_components / n_clusters
    print("Number of connected components: ", con_component_ratio)

    # Output the results
    for cluster, is_connected in connected_clusters.items():
        print(
            f"Cluster {cluster} is {'connected' if is_connected else 'not connected'} in the original network.")

    FQ = utils.compute_fast_modularity(clusters, num_nodes, num_edges, torch_sparse_adj, degree, device) # Compute modularity of the obtained community structure
    print('------------Testing Performance--------------------')
    print('No of clusters: ', max(clusters) + 1)
    print('Modularity:', FQ)

    NMI_original = utils.compute_nmi(clusters, data.y.squeeze().cpu().numpy()) # Compute NMI
    # NMI_processed_label = utils.compute_nmi(clusters, labels_modified)
    print('NMI_original:', NMI_original)
    # print('NMI_processed_label:', NMI_processed_label)

    conductance = utils.compute_conductance(clusters, Graph) # Compute conductance
    avg_conductance = sum(conductance) / len(conductance)
    print('Conductance: ', avg_conductance * 100)

    f1_score = utils.sample_f1_score(test_data, clusters, num_nodes) # Compute f1_score
    print('Sample_F1_score:', f1_score)
    print('-------------------------------------------------------')

    # con_Q = nx.algorithms.community.modularity(Graph, con_components)
    # NMI_processed_label = utils.compute_nmi(community_to_membership(con_components), labels_original)

    results = {
        'num_clusters': np.unique(clusters).shape[0],
        'modularity': FQ,
        'nmi': NMI_original,
        'conductance': avg_conductance,
        'sample_f1_score': f1_score
    }
    # results.append([run, FQ, NMI_original, NMI_processed_label, avg_conductance, f1_score, np.unique(clusters).shape[0]])

    if not os.path.exists('results'):
        os.makedirs('results')
    # csv_file = f'results/processedGT_basedonLeiden/{dataset_name}/results_{dataset_name}_{lam}_L_M_Performance.csv'
    # with open(csv_file, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(
    #         ['Run', 'Modularity', 'NMI_original', 'NMI_processed_label' 'Conductance', 'F1_score',
    #          'num_clusters'])
    #     writer.writerows(results)

# # csv_file = f'results/Leiden_30runs_lossL1/{dataset_name}/Weighted_LeidenCluster_loss/DGCluster/results_{dataset_name}_{lam}_DGCluster.csv'
            # csv_file = f'results/processedGT_basedonLeiden/{dataset_name}/results_{dataset_name}_{lam}_Performance.csv'
            # with open(csv_file, mode='w', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(['Run', 'Modularity', 'NMI_original', 'NMI_processed_label' 'Conductance', 'F1_score', 'num_clusters'])
            #     writer.writerows(results)


    if alp == 0.0:
        torch.save(results, f'results/results_{dataset_name}_{lam}_{epochs}_{base_model}_{seed}.pt')
    else:
        torch.save(results, f'results/results_{dataset_name}_{lam}_{alp}_{epochs}_{base_model}_{seed}.pt')
