import pandas as pd
from datetime import datetime, timedelta

def generate():
    df = pd.read_csv("SMT_2024/SMT_Algebra_2024.csv").rename(columns={"#": "id"})
    # big_df = df

    # sort by id
    df.sort_values(by=['id'], inplace=True)

    for i in range(1, 11):
        col = f"A{i}"
        correct_col = f"C{i}"
        freq_col = f"A{i}_freq"
        # Create a freq_col
        df[freq_col] = 1.0
        
        counts = df[df[correct_col] == 0][col].value_counts()
        df.loc[df[correct_col] == 0, freq_col] = df.loc[df[correct_col] == 0, col].map(counts/counts.sum())
        df.loc[df[correct_col] == 1, freq_col] = df[f"C{i}"].mean()



    # Get the frequency of each value in the 'A1' to 'A10' columns
    for i in range(1, 11):
        correct_col = f"C{i}"
        # correct_perc = f"C{i}_perc"

        df[correct_col] = df[correct_col].fillna(2)

        df[correct_col] = df[correct_col].astype(int)
        # df[correct_col] is in [0, 1, 2]. Create this into a 3 dimensional 1-hot encoding and add it to the dataframe
        one_hot = pd.get_dummies(df[correct_col], prefix=correct_col)
        one_hot = one_hot.astype(float)
        df = pd.concat([df, one_hot], axis=1)
        # # Create a freq_col
        # df[freq_col] = 0.0
        
        # counts = big_df[big_df[correct_col] == 0][col].value_counts()
        # df.loc[df[correct_col] == 0, freq_col] = df.loc[df[correct_col] == 0, col].map(counts/counts.sum())
        # df.loc[df[correct_col] == 1, freq_col] = df[f"C{i}"].mean()


    # for i in range(1, 11):
    #     correct_perc = f"C{i}_perc"
    #     df[correct_perc] = big_df[f"C{i}"].mean()

    time_df = pd.DataFrame()
    # Add the Start Time to the times in 'T1' to 'T10'
    for i, col in enumerate([f"T{i}" for i in range(1, 11)]):
        # Convert 'Start Time' to datetime  
        start_time = pd.to_datetime(df['Start Time'])
        
        # Convert duration in df[col] to timedelta
        duration = pd.to_timedelta(df[col])
        
        # Add duration to start time
        time_df[col] = start_time + duration
        # import pdb; pdb.set_trace()
        # time_df.loc[time_df[col].isna(), col] = time_df.loc[time_df[col].isna(), f"T{max(i, 1)}"]
    # Replace the nans with the value in the previous column
    # time_df.fillna(method='ffill', inplace=True, axis=1)
    for name, row in time_df.iterrows():
        time_df.loc[name, row.isna()] = row.max()
        # for i, col in enumerate([f"T{i}" for i in range(1, 11)]):
        #     if pd.isna(row[1][col]):
        #         row[1][col] = row[1][f"T{max(i, 1)}"]

    for col in [f"T{i}" for i in range(1, 11)]:
        # # Subtract the average time from the time in the column
        df[col] = (time_df[col] - time_df[col].mean()).dt.total_seconds()
        df[col] = df[col] / df[col].std()



    # Drop the 'Start Time' column
    df.drop(columns=['Start Time'], inplace=True)

    # Drop the 'Score' column
    df.drop(columns=['Score'], inplace=True)
    return df 

def make_torch_geometric(df):
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--big', action='store_true', help='Process the entire subset of the data')
    args = parser.parse_args()
    
    df = generate()
    if not args.big:
        # small_df = pd.read_csv("SMT_2024/SMT_Algebra_2024_Small.csv").rename(columns={"#": "id"})
        small_df = pd.read_csv("SMT_2024/SMT_Algebra_2024_processed_small.csv").rename(columns={"#": "id"})

        df = df[df['id'].isin(small_df['id'])]

        df.to_csv("./SMT_2024/SMT_Algebra_2024_Small_processed.csv", index=False)
    else:
        df.to_csv("./SMT_2024/SMT_Algebra_2024_processed.csv", index=False)


