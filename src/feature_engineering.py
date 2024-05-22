import pandas as pd

def feature_engineering(data):
    # Example: Creating a new feature "TotalSF" as the sum of all square footage features
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    
    # Drop features that are not useful
    # You should not drop the 'Id' column here
    # data = data.drop(['Id'], axis=1)  # Commented out to retain the 'Id' column
    
    return data
