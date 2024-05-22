import sys
import os

# add project directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pandas as pd
from src.data_processing import load_data, preprocess_data, split_data
from src.feature_engineering import feature_engineering
from src.model import train_random_forest, train_linear_regression, evaluate_model

# Load data
train_data = load_data('data/raw/train.csv')
test_data = load_data('data/raw/test.csv')

# Preprocess data
train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

X_train, y_train, X_test = preprocess_data(train_data, test_data)

# Split data
X_train, X_val, y_train, y_val = split_data(X_train, y_train)

# Train models
rf_model = train_random_forest(X_train, y_train)
lr_model = train_linear_regression(X_train, y_train)

# Evaluate models
rf_rmse = evaluate_model(rf_model, X_val, y_val)
lr_rmse = evaluate_model(lr_model, X_val, y_val)

print(f'Random Forest RMSE: {rf_rmse}')
print(f'Linear Regression RMSE: {lr_rmse}')

# Choose the best model for submission
best_model = rf_model if rf_rmse < lr_rmse else lr_model

# Make predictions
test_predictions = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

submission.to_csv('submission.csv', index=False)
