import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, RGCNConv, Linear
import time

# Create a fake heterogeneous graph dataset
data = HeteroData()

# Add node features for a single node type
data['node'].x = torch.randn(100, 16)  # 100 nodes with 16 features

# Add two types of edges
data['node', 'edge_type_1', 'node'].edge_index = torch.randint(0, 100, (2, 300))  # 300 edges
data['node', 'edge_type_2', 'node'].edge_index = torch.randint(0, 100, (2, 200))  # 200 edges

# Define global labels for every pair of nodes
num_node_pairs = 100 * 100  # Assume all node pairs
node_pair_labels = torch.zeros(num_node_pairs, dtype=torch.long)
i, j = torch.meshgrid(torch.arange(100), torch.arange(100), indexing='ij')

node_pair_labels = torch.sign((data['node'].x[i.flatten()] + data['node'].x[j.flatten()]).sum(dim=1)).long()
node_pair_labels[node_pair_labels == -1] = 0  # Convert -1 to 0 for binary classification

# 1. Heterogeneous GNN with GAT
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.conv = HeteroConv({
            edge_type: GATConv(16, 32, heads=2, concat=True) 
                for edge_type in metadata[1]
        }, aggr='sum')
        self.lin = Linear(32 * 2, 2)  # Output layer for pairwise classification 

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        node_embeddings = x_dict['node']
        return node_embeddings

# 2. Relational GCN
class RelationalGNN(torch.nn.Module):
    def __init__(self, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(16, 32, num_relations=num_relations)
        self.conv2 = RGCNConv(32, 16, num_relations=num_relations)
        self.lin = Linear(16, 2)  # Output layer for pairwise classification

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

# Instantiate models and datasets
metadata = data.metadata()
hetero_model = HeteroGNN(metadata)
relational_model = RelationalGNN(num_relations=2)

# Optimizer and loss function for both models
criterion = torch.nn.CrossEntropyLoss()
optimizer_hetero = torch.optim.Adam(hetero_model.parameters(), lr=0.01)
optimizer_relational = torch.optim.Adam(relational_model.parameters(), lr=0.01)


start = time.time()
# Training loop for HeteroGNN
for epoch in range(50):
    hetero_model.train()
    optimizer_hetero.zero_grad()
    embeddings = hetero_model(
        x_dict={'node': data['node'].x},
        edge_index_dict={
            ('node', 'edge_type_1', 'node'): data['node', 'edge_type_1', 'node'].edge_index,
            ('node', 'edge_type_2', 'node'): data['node', 'edge_type_2', 'node'].edge_index,
        },
    )
    # Pairwise predictions for all node pairs
    i, j = torch.meshgrid(torch.arange(100), torch.arange(100), indexing='ij')

    # node_pairs = torch.cat([embeddings[i.flatten()], embeddings[j.flatten()]], dim=1)
    node_pairs = embeddings[i.flatten()] + embeddings[j.flatten()]
    predictions = hetero_model.lin(node_pairs)
    loss = criterion(predictions, node_pair_labels)
    loss.backward()
    optimizer_hetero.step()
    print(f"HeteroGNN Epoch {epoch + 1}, Loss: {loss.item():.4f}")
print(f"Time taken: {time.time() - start:.2f}s")

start = time.time()
# Training loop for RelationalGNN
for epoch in range(50):
    relational_model.train()
    optimizer_relational.zero_grad()
    embeddings = relational_model(
        x=data['node'].x,
        edge_index=torch.cat(
            [data['node', 'edge_type_1', 'node'].edge_index, data['node', 'edge_type_2', 'node'].edge_index], dim=1
        ),
        edge_type=torch.cat(
            [
                torch.zeros(data['node', 'edge_type_1', 'node'].edge_index.size(1), dtype=torch.long),
                torch.ones(data['node', 'edge_type_2', 'node'].edge_index.size(1), dtype=torch.long),
            ]
        ),
    )
    # Pairwise predictions for all node pairs
    i, j = torch.meshgrid(torch.arange(100), torch.arange(100), indexing='ij')
    # node_pairs = torch.cat([embeddings[i.flatten()], embeddings[j.flatten()]], dim=1)
    node_pairs = embeddings[i.flatten()] + embeddings[j.flatten()]
    predictions = relational_model.lin(node_pairs)
    loss = criterion(predictions, node_pair_labels)
    loss.backward()
    optimizer_relational.step()
    print(f"RelationalGNN Epoch {epoch + 1}, Loss: {loss.item():.4f}")

print(f"Time taken: {time.time() - start:.2f}s")