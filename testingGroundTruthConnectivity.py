import leidenalg
import torch
from torch_geometric.datasets import Planetoid
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from cdlib import algorithms as cdalg
import networkx as nx
import numpy

# Step 1: Load the Cora Dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Step 2: Extract Ground Truth Community Labels
node_labels = data.y.numpy()  # Ground truth labels
node_indices = torch.arange(data.num_nodes).numpy()  # Node indices

# Step 3: Create a Graph Representation
# Convert the edge index to a NetworkX graph
G = nx.Graph()
edges = data.edge_index.numpy().T
G.add_edges_from(edges)

# Step 4: Organize nodes by their community labels
communities = defaultdict(list)
for idx, label in zip(node_indices, node_labels):
    communities[label].append(idx)

# Step 5: Find the community with the smallest size
smallest_community_label = min(communities, key=lambda x: len(communities[x]))
nodes_in_smallest_community = communities[smallest_community_label]

# Original ground truth modularity
com_list = list(communities.values())
# Q_GT_original = nx.algorithms.community.modularity(G, com_list)




def community_to_membership(individual):
    new_com_dict = {}
    for k in range(0, len(individual)):
        for element in individual[k]:
            new_com_dict[int(element)] = k+1
        # print("new community dictionary: ", new_com_dict)
    sort_com_dict = dict(sorted(new_com_dict.items()))
    # part_membership = [new_com_dict[key] for key in sort_com_dict]
    return sort_com_dict

# Function to process the ground truth using Leiden
def preprocess_groundtruth_Leiden(G, GTcommunities):
    # Sort the dictionary by keys
    sorted_GTcommunities = dict(sorted(GTcommunities.items()))
    leiden_communities = []
    leiden_GTcommunities = {}
    for key in sorted_GTcommunities:
        com_subgraph = G.subgraph(list(sorted_GTcommunities[key]))
        leiden_subgraph_com = cdalg.leiden(com_subgraph).communities
        for l in leiden_subgraph_com:
            leiden_communities.append(l)
        count = 0
        for nodes_list in leiden_communities:
            leiden_GTcommunities[count] = nodes_list
            count += 1
    return leiden_GTcommunities




# Function to visualize a particular community
def visualize_community(G, nodes, community_label):
    # Create a subgraph for the specified community
    subgraph = G.subgraph(nodes)

    # Draw the subgraph with nodes and edges
    pos = nx.spring_layout(subgraph, k=0.3)  # Layout for better visualization
    plt.figure(figsize=(5, 5))

    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, node_size=100, node_color='skyblue', label=None)

    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5, edge_color='gray')

    # Draw labels
    nx.draw_networkx_labels(subgraph, pos, font_size=5)

    # Display graph title and show plot
    plt.title(f'Community {community_label} Visualization with Edges')
    plt.axis('off')  # Turn off axis
    plt.show()

def is_connected_component(G, nodes):
    subgraph = G.subgraph(nodes)
    # leiden_communities = cdalg.leiden(subgraph).communities
    # count = 0
    # for com in leiden_communities:
    #     if nx.is_connected(G.subgraph(com)) == False:
    #         print('Leiden com is not connected')
    #         count += 1
    # if count == 0:
    #     print('All the Leiden communities are connected')
    # connected_components = list(nx.connected_components(subgraph))

    # Identify disconnected nodes
    # disconnected_nodes = []
    # for component in connected_components:
    #     if len(component) == 1:
    #         disconnected_nodes.extend(component)
    # print('disconnected nodes: ', disconnected_nodes)
    return nx.is_connected(subgraph)

# Processed ground truth
# com_leiden = cdalg.leiden(G, initial_membership=node_labels).communities
# Q_GT_processed = nx.algorithms.community.modularity(G, com_leiden)

# com_leiden_original_network = cdalg.leiden(G).communities
# Q_original_network = nx.algorithms.community.modularity(G, com_leiden_original_network)


GT_byLeiden = preprocess_groundtruth_Leiden(G, communities)
# Verify each community
# for community, nodes in communities.items():
for community, nodes in GT_byLeiden.items():
    if is_connected_component(G, nodes):
        print(f"Community '{community}' is a connected component.")
    else:
        print(f"Community '{community}' is NOT a connected component.")

print('Original network is connected:', nx.is_connected(G))

# Visualize the smallest community
# visualize_community(G, nodes_in_smallest_community, smallest_community_label)
