import pandas as pd
import numpy as np

df = pd.read_csv('Feature_Reduced_Set.csv')

# drop columns that are not needed for modeling
df = df.drop(columns=['home_win', 'neutral'])

# sort chronologically within each team
df = df.sort_values(['home_season', 'week']).reset_index(drop=True)

# separate team and year from home_season
df['team'] = df['home_season'].apply(lambda x: x.split('_')[0])
df['year'] = df['home_season'].apply(lambda x: int(x.split('_')[1]))

train = df[df['year'] <= 2022]
val   = df[df['year'] == 2023]
test  = df[df['year'] == 2024]

# function to get sequences of features and targets for a given window size
def build_sequences(data, window_size, feature_cols, target_col='spread'):
    X, y = [], []
    
    # group by team so sequences don't bleed across teams
    for team, group in data.groupby('team'):
        group = group.sort_values('week').reset_index(drop=True)
        
        # need at least window_size + 1 games to make sequences
        if len(group) < window_size + 1:
            continue
        
        for i in range(len(group) - window_size):
            seq = group[feature_cols].iloc[i:i+window_size].values
            target = group[target_col].iloc[i+window_size]
            
            X.append(seq)
            y.append(target)
    
    # Convert to float arrays before returning
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return X, y

# defines the feature columns 
feature_cols = [
    'week', 'rank_away', 'rank_home',
    'q1_home', 'q2_home', 'q3_home', 'q4_home', 
    'first_downs_home', 'fum_home', 'int_home', 
    'pen_num_home', 'pen_yards_home', 'possession_home',
    'Home Third Down Completion Rate', 'Home Pass Completion Rate', 
    'Run-Pass Ratio', 'Rush Yards - Total Yards'
]

# Remove any columns that couldn't be converted
for col in feature_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop columns that are still object (string) type
numeric_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']]
feature_cols = numeric_cols

df = df.dropna()

# builds sequences for each window size
for window in [3, 4, 5]:
    X_train, y_train = build_sequences(train, window, feature_cols)
    X_val,   y_val   = build_sequences(val,   window, feature_cols)
    X_test,  y_test  = build_sequences(test,  window, feature_cols)
    
    #print(f"Window {window}:")
    #print(f"  Train: {X_train.shape}, {y_train.shape}")
    #print(f"  Val:   {X_val.shape}, {y_val.shape}")
    #print(f"  Test:  {X_test.shape}, {y_test.shape}")