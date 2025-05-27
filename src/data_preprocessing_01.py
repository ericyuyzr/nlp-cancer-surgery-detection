import pandas as pd
from sklearn.model_selection import train_test_split

# Load your CSV file
df = pd.read_csv('data/diagnosis_procedure_list.csv')  # Replace with your actual file path
df = df.rename(columns={'site': 'target'})

# Prepare empty dataframes for train and test sets
train_df = pd.DataFrame()
test_df = pd.DataFrame()

# Split each site separately
for site in df['target'].unique():
    site_data = df[df['target'] == site]
    site_train, site_test = train_test_split(
        site_data, 
        test_size=0.2, 
        random_state=42,  # for reproducibility
        shuffle=True
    )
    train_df = pd.concat([train_df, site_train], ignore_index=True)
    test_df = pd.concat([test_df, site_test], ignore_index=True)

# Optionally shuffle the final train and test sets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new CSVs
train_df.to_csv('data/train_data_01.csv', index=False)
test_df.to_csv('data/test_data_01.csv', index=False)