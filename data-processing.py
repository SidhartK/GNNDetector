import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv("SMT_2024/SMT_Algebra_2024.csv").rename(columns={"#": "id"})

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
    df.loc[df[correct_col] == 1, freq_col] = df[f"C{i}"].mean()

for i in range(1, 11):
    correct_perc = f"C{i}_perc"
    df[correct_perc] = big_df[f"C{i}"].mean()


# Add the Start Time to the times in 'T1' to 'T10'
for col in [f"T{i}" for i in range(1, 11)]:
    # Convert 'Start Time' to datetime
    start_time = pd.to_datetime(df['Start Time'])
    
    # Convert duration in df[col] to timedelta
    duration = pd.to_timedelta(df[col])
    
    # Add duration to start time
    df[col] = start_time + duration

# Drop the 'Start Time' column
df.drop(columns=['Start Time'], inplace=True)

# Drop the 'Score' column
df.drop(columns=['Score'], inplace=True)


df.to_csv("./SMT_2024/SMT_Algebra_2024_processed.csv", index=False)