import numpy as np

def calculate_standard_score(y_true, y_pred) -> float:
    """
    Contracted Brain: Calculate a scalar score for Regression or Classification.
    Input: y_true (numpy), y_pred (numpy).
    Output: Return a float. Standard is 'Lower is Better' (e.g. MSE, 1-Accuracy).
    """
    # AI: Implement specific scoring logic based on task background
    mse = np.mean((y_true - y_pred)**2)
    return float(mse)
