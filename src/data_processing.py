import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(train_df, test_df):
    # Concatenate train and test data
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Handle missing values
    numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    combined_df[numeric_cols] = imputer.fit_transform(combined_df[numeric_cols])

    # One-hot encoding
    combined_df = pd.get_dummies(combined_df)

    # Split train and test data
    train_processed = combined_df[:len(train_df)]
    test_processed = combined_df[len(train_df):]

    return train_processed, test_processed
