import pandas as pd

# Load the training dataset
train_df = pd.read_csv('data/raw/train.csv')

# Display the first few rows of the training dataset
print("Training Dataset:")
print(train_df.head())

# Display basic information about the training dataset
print("\nInfo about Training Dataset:")
print(train_df.info())

# Check for missing values in the training dataset
print("\nMissing Values in Training Dataset:")
print(train_df.isnull().sum())
