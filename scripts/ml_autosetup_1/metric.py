import numpy as np
from sklearn.metrics import mean_squared_error

# --- Metric Configuration ---
# Set this to True if the metric is like Accuracy (higher is better).
# Set this to False if the metric is like MSE/Loss (lower is better).
HIGHER_IS_BETTER = False 

def calculate_score(y_true, y_pred):
    """
    Calculates the score for the given predictions.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values from the strategy.
        
    Returns:
        float: The calculated score.
    """
    # Example: Mean Squared Error for Regression
    # Since inputs might be (N, 1000), this calculates the average MSE across all dimensions.
    return mean_squared_error(y_true, y_pred)
