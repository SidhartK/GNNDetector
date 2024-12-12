import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from tqdm import tqdm  # Import tqdm for progress bar

import matplotlib.pyplot as plt
from scipy.stats import norm


def cheating_probability(row1, row2, k1=0.5, k2=0.7, k3=0.6, t_thresh=10, verbose = False):
    w1, w2, w3 = 0.2, 0.5, 0.3
    # Correct/Incorrect Answer Matching (CIM)
    num_data_points = 10
    cim_matches = sum([row1[f'C{i}'] == row2[f'C{i}'] for i in range(1, num_data_points + 1)])
    p_cim = cim_matches**2/num_data_points**2
    
    # Matching Incorrect Answers (MIA)
    mia_matches = 0
    for i in range(1, num_data_points + 1):
        if row1[f'C{i}'] == 0 and row2[f'C{i}'] == 0 and row1[f'A{i}'] == row2[f'A{i}']:
            rarity_weight = 1 - row1[f'A{i}_freq']  # Higher weight for rarer answers
            mia_matches += rarity_weight
    p_mia = 1 - np.exp(-0.8 * (mia_matches ** 2))
    #print("p_mia:", p_mia)
    
    # Timing Similarity (TS)
    ts_matches = 0
    for i in range(1, num_data_points + 1):
        # Check if either time is blank
        if not pd.isna(row1[f'T{i}']) and not pd.isna(row2[f'T{i}']):
            # Updated time format to YYYY/MM/DD HH:MM:SS
            if abs((pd.to_datetime(row1[f'T{i}']) - pd.to_datetime(row2[f'T{i}'])).total_seconds()) <= t_thresh:
                ts_matches += 1
    p_ts = 1 - np.exp(-k3 * ts_matches)
    
    # Combine probabilities
    cheating_prob = p_cim * w1 + p_mia * w2 + p_ts * w3
    if verbose:
        print("User1:", row1['id'], "User2:", row2['id'])
        print("CIM matches:", cim_matches)
        print("p_cim:", p_cim)
        print("MIA matches:", mia_matches)
        print("p_mia:", p_mia)
        print("TS matches:", ts_matches)
        print("p_ts:", p_ts)
        print("Cheating probability:", cheating_prob)
    return cheating_prob

def debug_cheating_probability(df, user1_id, user2_id):
    # Find the rows corresponding to the given user IDs
    row1 = df[df['id'] == user1_id].iloc[0]
    row2 = df[df['id'] == user2_id].iloc[0]
    
    # Call cheating_probability with the found rows
    return cheating_probability(row1, row2, verbose=True)

def cheating_probability_matrix(df):
    num_students = df.shape[0]
    matrix = np.zeros((num_students, num_students))  # Initialize a matrix to store probabilities

    for i in tqdm(range(num_students), desc="Calculating cheating probabilities"):
        for j in range(i+1, num_students):
            matrix[i, j] = cheating_probability(df.iloc[i], df.iloc[j])  # Calculate cheating probability for each pair

    # New code to find and print top 5 and bottom 5 pairs
    cheating_pairs = []
    for i in range(num_students):
        for j in range(i+1, num_students):
            cheating_pairs.append((i, j, matrix[i, j]))

    # Sort pairs by cheating probability
    cheating_pairs.sort(key=lambda x: x[2], reverse=True)

    print("Top 5 pairs with highest cheating probability:")
    count = 0
    for i in range(len(cheating_pairs)):
        if cheating_pairs[i][0] < cheating_pairs[i][1]:  # Ensure the first index is less than the second
            print(f"User {df.iloc[cheating_pairs[i][0]]['id']} and User {df.iloc[cheating_pairs[i][1]]['id']} : {cheating_pairs[i][2]}")
            count += 1
        if count == 50:  # Stop after printing 5 valid pairs
            break
    '''
    print("\nBottom 5 pairs with lowest cheating probability:")
    count = 0
    for i in range(1, len(cheating_pairs) + 1):
        if cheating_pairs[-i][0] < cheating_pairs[-i][1]:  # Ensure the first index is less than the second
            print(f"User {cheating_pairs[-i][0]} and User {cheating_pairs[-i][1]}: {cheating_pairs[-i][2]}")
            count += 1
        if count == 5:  # Stop after printing 5 valid pairs
            break
    '''
    return matrix

if __name__ == '__main__':
    # Main execution
    df = pd.read_csv('SMT_2024/SMT_Algebra_2024_Small_processed.csv')
    # debug_cheating_probability(df, '069D', '069D')
    cheating_matrix = cheating_probability_matrix(df)  # Get the cheating probability matrix
    # import pdb; pdb.set_trace()
    # Generate some data
    data = cheating_matrix.flatten()

    # Create a CDF using norm.cdf()
    x = np.sort(data)
    y = norm.cdf(x, np.mean(data), np.std(data))

    # Plot the CDF
    plt.plot(x, y)
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.title('CDF of a Heuristic Distribution')
    plt.show()
    #print("Cheating Probability Matrix:") 
    #print(cheating_matrix)
