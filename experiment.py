import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, RGCNConv, Linear
import time

import pickle
import optuna

"""
0. Get data
- Run `python data-processing.py`
- Run `python torch-geo-dataset.py`
"""

# 1. Heterogeneous GNN with GAT
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels=128):
        super().__init__()
        self.conv = HeteroConv({
            edge_type: GATConv(in_channels, hidden_channels, heads=2, concat=True) 
                for edge_type in metadata[1]
        }, aggr='sum')
        self.lin = Linear(hidden_channels * 2, 1)  # Output layer for pairwise classification 

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        node_embeddings = x_dict['node']
        return node_embeddings

# 2. Relational GCN
class RelationalGNN(torch.nn.Module):
    def __init__(self, num_relations, in_channels, hidden_channels=[256, 64]):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels[0], num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels[0], hidden_channels[0], num_relations=num_relations)
        self.conv3 = RGCNConv(hidden_channels[0], hidden_channels[0], num_relations=num_relations)
        self.conv4 = RGCNConv(hidden_channels[0], hidden_channels[1], num_relations=num_relations)
        self.lin = Linear(hidden_channels[1], 1)  # Output layer for pairwise classification

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_type)
        return x


with open("data/data.pkl", "rb") as f:
    data, labels = pickle.load(f)

num_nodes = data['node'].x.size(0)      # 136
in_channels = data['node'].x.size(1)    # 50: (10 * 3) + 10 + 10
# Instantiate models and datasets
metadata = data.metadata()
hetero_model = HeteroGNN(metadata, in_channels=in_channels)
relational_model = RelationalGNN(num_relations=len(metadata[1]), in_channels=in_channels)

def get_predictions(embeddings, model):
# Pairwise predictions for all node pairs
    i, j = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')

    # node_pairs = torch.cat([embeddings[i.flatten()], embeddings[j.flatten()]], dim=1)
    node_pairs = embeddings[i.flatten()] + embeddings[j.flatten()]
    return model.lin(node_pairs)

# Binary cross-entropy loss (https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
criterion = torch.nn.BCEWithLogitsLoss()
def loss_func(pred, target, reg_param=0.1):
    heuristic_rec_loss = criterion(pred.flatten(), target[:, 0])
    team_rec_loss = criterion(pred.flatten(), target[:, 1])
    return (heuristic_rec_loss * reg_param) + team_rec_loss
    # return team_rec_loss
    

# Optimizer and loss function for both models
optimizer_hetero = torch.optim.Adam(hetero_model.parameters(), lr=0.01)
optimizer_relational = torch.optim.Adam(relational_model.parameters(), lr=0.01)


# start = time.time()
# # Training loop for HeteroGNN
# for epoch in range(200):
#     hetero_model.train()
#     optimizer_hetero.zero_grad()
#     embeddings = hetero_model(
#         x_dict={'node': data['node'].x},
#         edge_index_dict={
#             edge_type: data[edge_type].edge_index
#                 for edge_type in metadata[1]
#         },
#     )
#     predictions = get_predictions(embeddings, hetero_model)
#     # # Pairwise predictions for all node pairs
#     # i, j = torch.meshgrid(torch.arange(100), torch.arange(100), indexing='ij')

#     # # node_pairs = torch.cat([embeddings[i.flatten()], embeddings[j.flatten()]], dim=1)
#     # node_pairs = embeddings[i.flatten()] + embeddings[j.flatten()]
#     # predictions = hetero_model.lin(node_pairs)
#     # loss = criterion(predictions, node_pair_labels)
#     loss = loss_func(predictions, labels)
#     loss.backward()
#     optimizer_hetero.step()
#     print(f"HeteroGNN Epoch {epoch + 1}, Loss: {loss.item():.4f}")
# print(f"Time taken: {time.time() - start:.2f}s")

start = time.time()
# Training loop for RelationalGNN
for epoch in range(500):
    relational_model.train()
    optimizer_relational.zero_grad()
    embeddings = relational_model(
        x=data['node'].x,
        edge_index=torch.cat(
            [data[edge_type].edge_index for edge_type in metadata[1]], dim=1
        ),
        edge_type=torch.cat(
            [
                torch.full((data[edge_type].edge_index.size(1),), i, dtype=torch.long)
                    for i, edge_type in enumerate(metadata[1])
            ]
        ),
    )
    # Pairwise predictions for all node pairs
    # i, j = torch.meshgrid(torch.arange(100), torch.arange(100), indexing='ij')
    # # node_pairs = torch.cat([embeddings[i.flatten()], embeddings[j.flatten()]], dim=1)
    # node_pairs = embeddings[i.flatten()] + embeddings[j.flatten()]
    # predictions = relational_model.lin(node_pairs)
    predictions = get_predictions(embeddings, relational_model)
    loss = loss_func(predictions, labels)
    loss.backward()
    optimizer_relational.step()
    print(f"RelationalGNN Epoch {epoch + 1}, Loss: {loss.item():.4f}")

print(f"Time taken: {time.time() - start:.2f}s")