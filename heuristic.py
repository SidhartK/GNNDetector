import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm  # Import tqdm for progress bar

def cheating_probability(row1, row2, k1=0.5, k2=0.7, k3=0.6, t_thresh=10):
    # Correct/Incorrect Answer Matching (CIM)
    num_data_points = 10
    cim_matches = sum([row1[f'C{i}'] == row2[f'C{i}'] for i in range(1, num_data_points + 1)])
    #print("CIM matches:", cim_matches)
    p_cim = cim_matches**2/num_data_points**2
    #print("p_cim:", p_cim)
    
    # Matching Incorrect Answers (MIA)
    mia_matches = sum([
        row1[f'C{i}'] == 0 and row2[f'C{i}'] == 0 and row1[f'A{i}'] == row2[f'A{i}']
        for i in range(1, num_data_points + 1)
    ])
    p_mia = 1 - np.exp(-k2 * mia_matches)
    #print("p_mia:", p_mia)
    
    # Timing Similarity (TS)
    ts_matches = 0
    for i in range(1, num_data_points + 1):
        # Check if either time is blank
        #print(row1[f'T{i}'], row2[f'T{i}'])
        if not pd.isna(row1[f'T{i}']) and not pd.isna(row2[f'T{i}']):
            if abs((datetime.strptime(row1[f'T{i}'], '%H:%M:%S') - datetime.strptime(row2[f'T{i}'], '%H:%M:%S')).total_seconds()) <= t_thresh:
                ts_matches += 1
    p_ts = 1 - np.exp(-k3 * ts_matches)
    
    # Combine probabilities
    cheating_prob = 1 - (1 - p_cim) * (1 - p_mia) * (1 - p_ts)
    return cheating_prob

def cheating_probability_matrix(df):
    num_students = df.shape[0]
    matrix = np.zeros((num_students, num_students))  # Initialize a matrix to store probabilities

    for i in tqdm(range(num_students), desc="Calculating cheating probabilities"):
        for j in range(num_students):
            matrix[i, j] = cheating_probability(df.iloc[i], df.iloc[j])  # Calculate cheating probability for each pair

    return matrix

# Main execution
df = pd.read_csv('SMT_2024/SMT_Algebra_2024_Small.csv')
cheating_matrix = cheating_probability_matrix(df)  # Get the cheating probability matrix
print("Cheating Probability Matrix:")
print(cheating_matrix)