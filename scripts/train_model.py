import sys
import os

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.data_processing import load_data, preprocess_data, split_data
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_processing import load_data, preprocess_data, split_data
from src.feature_engineering import feature_engineering
from src.model import train_linear_regression, train_random_forest, evaluate_model

# Load data
train_data = load_data('data/raw/train.csv')
test_data = load_data('data/raw/test.csv')

# Preprocess data
X_train, y_train, X_test = preprocess_data(train_data, test_data)  # Ensure this returns two values

# No need to call feature_engineering on preprocessed data, unless you have additional features to create

# Split the data
X_train, X_val, y_train, y_val = split_data(X_train, y_train)

# Train models
lr_model = train_linear_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Evaluate models
lr_rmse = evaluate_model(lr_model, X_val, y_val)
rf_rmse = evaluate_model(rf_model, X_val, y_val)

print(f'Linear Regression RMSE: {lr_rmse}')
print(f'Random Forest RMSE: {rf_rmse}')

# Make predictions
test_predictions = rf_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'Id': test_data['Id'].astype(int),
    'SalePrice': test_predictions
})

submission.to_csv('submission.csv', index=False)
