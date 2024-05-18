from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions,))
    return rmse