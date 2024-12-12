import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import pandas as pd
from heuristic import cheating_probability

import torch.nn.functional as F

# Create a simple graph dataset with edge attributes
def create_fake_dataset():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    edge_attr = torch.tensor([[1], [1], [1], [1]], dtype=torch.float)
    y = torch.tensor([0, 1, 0], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return [data]

def create_dataset():
    df = pd.read_csv("SMT_2024/SMT_Algebra_2024_Small_processed.csv").rename(columns={"#": "id"})
    edge_index = []
    edge_attr = []
    num_students = df.shape[0]
    for i in range(num_students):
        for j in range(i+1, num_students):
            edge_index.append([i, j])
            row_i, row_j = df.iloc[i], df.iloc[j]
            cp = cheating_probability(row_i, row_j)
            import pdb; pdb.set_trace()
            answer_match = (row_i[[f"A{k}" for k in range(1, 11)]] == row_j[[f"A{k}" for k in range(1, 11)]]).to_numpy()
            both_incorrect = (row_i[[f"C{k}" for k in range(1, 11)]] == 0) & (row_j[[f"C{k}" for k in range(1, 11)]] == 0).to_numpy()
            answer_likelihood = row_i[[f"A{k}_freq" for k in range(1, 11)]].to_numpy()
            edge_x = answer_match * both_incorrect * answer_likelihood

            
            edge_attr.append(np.concatenate([np.array([cp]), edge_x]))
    





# Define a simple GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train the model
def train(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# Evaluate the model
def evaluate(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# Main function
def main():
    dataset = create_dataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(20):
        train(model, loader, optimizer, criterion)
        acc = evaluate(model, loader)
        print(f'Epoch {epoch+1}, Accuracy: {acc:.4f}')

if __name__ == "__main__":
    main()