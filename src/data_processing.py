import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(train_data, test_data):
    # Separate features and target variable
    X_train = train_data.drop(columns=['Id', 'SalePrice'])
    y_train = train_data['SalePrice']
    X_test = test_data.drop(columns=['Id'])

    # Impute missing values
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    X_train[numerical_cols] = numerical_imputer.fit_transform(X_train[numerical_cols])
    X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
    X_test[numerical_cols] = numerical_imputer.transform(X_test[numerical_cols])
    X_test[categorical_cols] = categorical_imputer.transform(X_test[categorical_cols])

    # One-hot encode categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]))

    X_train_encoded.index = X_train.index
    X_test_encoded.index = X_test.index

    X_train.drop(columns=categorical_cols, inplace=True)
    X_test.drop(columns=categorical_cols, inplace=True)

    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, y_train, X_test

def split_data(X_train, y_train):
    return train_test_split(X_train, y_train, test_size=0.2, random_state=42)
