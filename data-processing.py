import pandas as pd

df = pd.read_csv("./SMT_2024/SMT_Algebra_2024_Small.csv").rename(columns={"#": "id"})
big_df = pd.read_csv("./SMT_2024/SMT_Algebra_2024.csv").rename(columns={"#": "id"})

# sort by id
df.sort_values(by=['id'], inplace=True)

# Get the frequency of each value in the 'A1' to 'A10' columns
for i in range(1, 11):
    col = f"A{i}"
    correct_col = f"C{i}"
    freq_col = f"A{i}_freq"
    correct_perc = f"C{i}_perc"
    # Create a freq_col
    df[freq_col] = 0.0
    
    counts = big_df[big_df[correct_col] == 0][col].value_counts()
    df.loc[df[correct_col] == 0, freq_col] = df.loc[df[correct_col] == 0, col].map(counts/counts.sum())
    df.loc[df[correct_col] == 1, freq_col] = 0.0

for i in range(1, 11):
    correct_perc = f"C{i}_perc"
    df[correct_perc] = big_df[f"C{i}"].mean()


# Add the Start Time to the times in 'T1' to 'T10'
for col in [f"T{i}" for i in range(1, 11)]:
    df[col] = df[col] + df['Start Time']

# Drop the 'Start Time' column
df.drop(columns=['Start Time'], inplace=True)

# Drop the 'Score' column
df.drop(columns=['Score'], inplace=True)


df.to_csv("./SMT_2024/SMT_Algebra_2024_Small_processed.csv", index=False)