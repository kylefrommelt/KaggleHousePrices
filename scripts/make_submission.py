import sys
import os

# add project directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pandas as pd
from src.data_processing import load_data, preprocess_data
from src.feature_engineering import feature_engineering
from src.model import train_random_forest

# load data
train_data = load_data('data/raw/train.csv')
test_data = load_data('data/raw/test.csv')

# preprocess data
train_data_processed, test_data_processed = preprocess_data(train_data, test_data)

# feature engineering
train_data_processed = feature_engineering(train_data_processed)
test_data_processed = feature_engineering(test_data_processed)

# prepare features for training
X_train = train_data_processed.drop('SalePrice', axis=1)
y_train = train_data_processed['SalePrice']

# train model
model = train_random_forest(X_train, y_train)

# make predictions
test_predictions = model.predict(test_data_processed.drop('SalePrice', axis=1))  # Exclude 'SalePrice' column

# create submission file
submission = pd.DataFrame({
    'Id': test_data_processed['Id'].astype(int),
    'SalePrice': test_predictions
})

submission.to_csv('submission.csv', index=False)
