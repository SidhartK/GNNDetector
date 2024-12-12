import numpy as np
import torch
from torch_geometric.data import HeteroData
import pandas as pd
from heuristic import cheating_probability

import torch.nn.functional as F


def create_dataset():
    df = pd.read_csv("SMT_2024/SMT_Algebra_2024_Small_processed.csv").rename(columns={"#": "id"})
    data = HeteroData()
    data["node"].x = torch.tensor(df[[f"C{k}" for k in range(1, 11)] + [f"A{k}_freq" for k in range(1, 11)] + [f"T{k}" for k in range(1, 11)]].to_numpy(), dtype=torch.float)
    edge_index = []
    edge_attr = []
    num_students = df.shape[0]
    for i in range(num_students):
        for j in range(i+1, num_students):
            edge_index.append([i, j])
            row_i, row_j = df.iloc[i], df.iloc[j]
            cp = cheating_probability(row_i, row_j)
            # answer_match = (row_i[[f"A{k}" for k in range(1, 11)]] == row_j[[f"A{k}" for k in range(1, 11)]]).to_numpy()
            # both_incorrect = (row_i[[f"C{k}" for k in range(1, 11)]] == 0) & (row_j[[f"C{k}" for k in range(1, 11)]] == 0).to_numpy()
            # answer_likelihood = 1 - row_i[[f"A{k}_freq" for k in range(1, 11)]].to_numpy()
            # time_difference = np.abs(row_i[[f"T{k}" for k in range(1, 11)]].to_numpy() - row_j[[f"T{k}" for k in range(1, 11)]].to_numpy())
            # answer_suss = answer_match * both_incorrect * answer_likelihood
            # edge_attr.append(np.concatenate([np.array([cp]), answer_suss, time_difference]))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.tensor(df[[f"C{k}" for k in range(1, 11)] + [f"A{k}_freq" for k in range(1, 11)]].to_numpy(), dtype=torch.float)
    


if __name__ == '__main__':
    create_dataset()