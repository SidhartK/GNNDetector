import numpy as np
import torch
from torch_geometric.data import HeteroData
import pandas as pd
from tqdm import trange
from heuristic import cheating_probability
import os
import pickle


def create_dataset():
    df = pd.read_csv("SMT_2024/SMT_Algebra_2024_Small_processed.csv").rename(columns={"#": "id"})
    data = HeteroData()

    data["node"].x = torch.tensor(df[[f"C{k}_{l}" for k in range(1, 11) for l in range(3)] + [f"A{k}_freq" for k in range(1, 11)] + [f"T{k}" for k in range(1, 11)]].to_numpy(), dtype=torch.float)
    num_students = df.shape[0]
    heuristic_labels = np.zeros((num_students, num_students))
    team_labels = np.zeros((num_students, num_students))
    for k in range(1, 11):
        data['node', f'edge_type_correct_{k}', 'node'].edge_index = []
        data['node', f'edge_type_incorrect_{k}', 'node'].edge_index = []

    for i in trange(num_students):
        for j in range(i+1, num_students):
            row_i, row_j = df.iloc[i], df.iloc[j]
            cp = cheating_probability(row_i, row_j)
            heuristic_labels[i, j] = cp
            heuristic_labels[j, i] = cp
            team_labels[i, j] = (row_i['id'][:-1] == row_j['id'][:-1])
            team_labels[j, i] = (row_i['id'][:-1] == row_j['id'][:-1])
            
            for k in range(1, 11):
                if (row_i[f"C{k}"] == 0.0) and (row_j[f"C{k}"] == 0.0) and (row_i[f"A{k}"] == row_j[f"A{k}"]):
                    data['node', f'edge_type_incorrect_{k}', 'node'].edge_index.extend([[i, j], [j, i]])
                elif (row_i[f"C{k}"] == 1.0) and (row_j[f"C{k}"] == 1.0):
                    data['node', f'edge_type_correct_{k}', 'node'].edge_index.extend([[i, j], [j, i]])

    for k in range(1, 11):
        if len(data['node', f'edge_type_correct_{k}', 'node'].edge_index) == 0:
            data['node', f'edge_type_correct_{k}', 'node'].edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            data['node', f'edge_type_correct_{k}', 'node'].edge_index = torch.tensor(data['node', f'edge_type_correct_{k}', 'node'].edge_index, dtype=torch.long).t().contiguous()
        
        if len(data['node', f'edge_type_incorrect_{k}', 'node'].edge_index) == 0:
            data['node', f'edge_type_incorrect_{k}', 'node'].edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            data['node', f'edge_type_incorrect_{k}', 'node'].edge_index = torch.tensor(data['node', f'edge_type_incorrect_{k}', 'node'].edge_index, dtype=torch.long).t().contiguous()

    return data, torch.stack([torch.tensor(heuristic_labels.flatten()), torch.tensor(team_labels.flatten())], dim=1)
if __name__ == '__main__':
    data, labels = create_dataset()

    if not os.path.exists("data/"):
        os.makedirs("data/")
    with open("data/data.pkl", "wb") as f:
        pickle.dump((data, labels), f)