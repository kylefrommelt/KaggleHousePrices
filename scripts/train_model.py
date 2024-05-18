import sys
import os
# add project directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_processing import load_data, preprocess_data
from src.feature_engineering import feature_engineering
from src.model import train_linear_regression, train_random_forest
from src.evaluation import evaluate_model

import numpy as np
from sklearn.metrics import mean_squared_error

# load data
train_data = load_data('data/raw/train.csv')
test_data = load_data('data/raw/test.csv')  # Load test data

# preprocess data
train_data, test_data = preprocess_data(train_data, test_data)  # Pass both train and test data

# feature engineering
train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

# check for NaN values
print("NaN values in train_data:", train_data.isna().sum().sum())

# prep features and target
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

# print shape of features and target to verify correct splitting
print(X.shape, y.shape)  # Should print matching number of rows

# train test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)

# train models
lr_model = train_linear_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# evaluate models
lr_rmse = evaluate_model(lr_model, X_val, y_val)
rf_rmse = evaluate_model(lr_model, X_val, y_val)

print(f'linear regression RMSE: {lr_rmse}')
print(f'random forest RMSE: {rf_rmse}')

# debugging: Check model predictions
lr_predictions = lr_model.predict(X_val)
rf_predictions = rf_model.predict(X_val)
print(f'Linear Regression Predictions: {lr_predictions[:5]}')
print(f'Random Forest Predictions: {rf_predictions[:5]}')

# verify if predictions are different
print(f'Difference in predictions: {sum(lr_predictions != rf_predictions)}')

# Print first 5 true values for comparison
print(f'True Values: {y_val.values[:5]}')

# Verify if predictions are different
print(f'Difference in predictions: {sum(lr_predictions != rf_predictions)}')

# Additional check: Print RMSE calculation components
print(f'Linear Regression True vs Predicted: {list(zip(y_val.values[:5], lr_predictions[:5]))}')
print(f'Random Forest True vs Predicted: {list(zip(y_val.values[:5], rf_predictions[:5]))}')

# Manual RMSE calculation for first 5 predictions
lr_rmse_manual = np.sqrt(mean_squared_error(y_val.values[:5], lr_predictions[:5]))
rf_rmse_manual = np.sqrt(mean_squared_error(y_val.values[:5], rf_predictions[:5]))

print(f'Manual Linear Regression RMSE (first 5): {lr_rmse_manual}')
print(f'Manual Random Forest RMSE (first 5): {rf_rmse_manual}')

# Manual RMSE calculation for all predictions
lr_rmse_full_manual = np.sqrt(mean_squared_error(y_val, lr_predictions))
rf_rmse_full_manual = np.sqrt(mean_squared_error(y_val, rf_predictions))

print(f'Manual Linear Regression RMSE (full): {lr_rmse_full_manual}')
print(f'Manual Random Forest RMSE (full): {rf_rmse_full_manual}')
