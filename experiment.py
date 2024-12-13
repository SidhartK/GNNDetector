import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, RGCNConv, Linear
import time
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


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
class RelationalGCN(torch.nn.Module):
    def __init__(self, num_relations, in_channels, hidden_channels=[256, 64]):
        """
        Parameters
        ----------
        num_relations : int
            Number of relation types between nodes.
        in_channels : int
            Number of features per node.
        hidden_channels : list of int, optional
            Number of hidden channels in each layer, by default [256, 64]
        """
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels[0], num_relations=num_relations)
        # self.conv2 = RGCNConv(hidden_channels[0], hidden_channels[0], num_relations=num_relations)
        # self.conv3 = RGCNConv(hidden_channels[0], hidden_channels[0], num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels[0], hidden_channels[1], num_relations=num_relations)
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
relational_model = RelationalGCN(num_relations=len(metadata[1]), in_channels=in_channels)

def evaluate_model(model, data, labels, loss_func, metadata):
    model.eval()
    with torch.no_grad():
        # Generate embeddings
        embeddings = model(
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
        # Generate predictions
        predictions = get_predictions(embeddings, model)
        # print("predictions", predictions)

        # Compute loss
        loss = loss_func(predictions, labels)
        
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions).flatten()
        
        # Threshold probabilities to obtain binary predictions
        preds_binary = (probs > 0.5).long()
        # print("sum_preds_binary", preds_binary.sum())
        
        # Flatten true labels
        true_labels = labels[:, 2].long()  # Use the third column for truth_labels
        # print("sum_true_labels", true_labels.sum())
        # Compute accuracy
        correct = (preds_binary == true_labels).sum().item()
        total = true_labels.size(0)
        # print("correct", correct, "total", total)
        accuracy = correct / total

        # Compute true positives, false positives, false negatives
        # print("preds_binary", preds_binary, "true_labels", true_labels)
        true_positive = ((preds_binary == 1) & (true_labels == 1)).sum().item()
        false_positive = ((preds_binary == 1) & (true_labels == 0)).sum().item()
        false_negative = ((preds_binary == 0) & (true_labels == 1)).sum().item()

        # Compute precision
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        
        # Compute recall
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        
        # Compute F1-score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute ROC-AUC (manual trapezoidal integration for simplicity)
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = true_labels[sorted_indices]
        sorted_probs = probs[sorted_indices]

        tpr = torch.cumsum(sorted_labels, dim=0) / sorted_labels.sum()
        fpr = torch.cumsum(1 - sorted_labels, dim=0) / (1 - sorted_labels).sum()
        roc_auc = torch.trapz(tpr, fpr).item() if tpr.numel() > 1 else 0.0

        return loss.item(), accuracy, precision, recall, f1, roc_auc


def get_predictions(embeddings, model):
# Pairwise predictions for all node pairs
    i, j = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')

    # node_pairs = torch.cat([embeddings[i.flatten()], embeddings[j.flatten()]], dim=1)
    node_pairs = embeddings[i.flatten()] + embeddings[j.flatten()]
    return model.lin(node_pairs)

# Calculate the number of positive and negative examples
num_positive = (labels[:, 2] == 1).sum().item()  # Count of positive examples
num_negative = (labels[:, 2] == 0).sum().item()  # Count of negative examples

# Calculate pos_weight as the ratio of negative to positive examples
pos_weight = num_negative / num_positive if num_positive > 0 else 1.0
# pos_weight = 12.0
# print("pos_weight", pos_weight)
# Binary cross-entropy loss with pos_weight
heuristic_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
team_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))
def train_loss_func(pred, target, reg_param=0.1):
    heuristic_rec_loss = heuristic_criterion(pred.flatten(), target[:, 0])
    team_rec_loss = team_criterion(pred.flatten(), target[:, 1])
    # print("heuristic_rec_loss", heuristic_rec_loss, "team_rec_loss", team_rec_loss)
    return (heuristic_rec_loss * reg_param) + team_rec_loss
    # return team_rec_loss
    # return heuristic_rec_loss

test_criterion = torch.nn.BCEWithLogitsLoss()
def test_loss_func(pred, target, reg_param=0.1):
    # heuristic_rec_loss = test_criterion(pred.flatten(), target[:, 0])
    # team_rec_loss = test_criterion(pred.flatten(), target[:, 1])
    label_rec_loss = test_criterion(pred.flatten(), target[:, 2])
    return label_rec_loss
    # return team_re1c_loss

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

heterogat = True
rgcn = True
num_epochs = 500
print_every = 50

if heterogat:
    print("-" * 50)
    start = time.time()
    # Training loop for HeteroGNN
    for epoch in range(num_epochs):
        verbose = (epoch + 1) % print_every == 0 
        hetero_model.train()
        optimizer_hetero.zero_grad()
        embeddings = hetero_model(
            x_dict={'node': data['node'].x},
            edge_index_dict={
                edge_type: data[edge_type].edge_index
                    for edge_type in metadata[1]
            },
        )
        predictions = get_predictions(embeddings, hetero_model)
        # # Pairwise predictions for all node pairs
        # i, j = torch.meshgrid(torch.arange(100), torch.arange(100), indexing='ij')

        # # node_pairs = torch.cat([embeddings[i.flatten()], embeddings[j.flatten()]], dim=1)
        # node_pairs = embeddings[i.flatten()] + embeddings[j.flatten()]
        # predictions = hetero_model.lin(node_pairs)
        # loss = criterion(predictions, node_pair_labels)
        loss = train_loss_func(predictions, labels)
        loss.backward()
        optimizer_hetero.step()
        if verbose:
            print(f"HeteroGAT Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            test_loss, accuracy, precision, recall, f1, roc_auc = evaluate_model(
                relational_model, data, labels, test_loss_func, metadata
            )
            print(f"Epoch {epoch + 1}, Test Loss: {test_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    print(f"Time taken: {time.time() - start:.2f}s")
    print("-" * 50)
if rgcn:
    print("-" * 50)
    start = time.time()
    # Training loop for RelationalGNN
    for epoch in range(num_epochs):

        verbose = (epoch + 1) % print_every == 0

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
        loss = train_loss_func(predictions, labels)
        loss.backward()
        optimizer_relational.step()
        if verbose:
            print(f"RelationalGNN Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            test_loss, accuracy, precision, recall, f1, roc_auc = evaluate_model(
                relational_model, data, labels, test_loss_func, metadata
            )
            print(f"Epoch {epoch + 1}, Test Loss: {test_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    print(f"Time taken: {time.time() - start:.2f}s")
    print("-" * 50)


