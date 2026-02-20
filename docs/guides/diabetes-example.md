# Diabetes Classification Example

This example demonstrates optimizing a Scikit-Learn model.

## Quick Start Guide

You can verify your installation immediately by running our pre-configured Scikit-Learn example without any setup.

1.  **Open Control Panel**: In the UI, stay on the **Control** tab.
2.  **Set Root Directory**: Set the `Root Directory` field to `example/diabetes_sklearn`.
3.  **Configure `eval_cmd`**:
    *   In the **Global Variables** list, find `eval_cmd`.
    *   Ensure it is set to: `python evaluator.py` (ensure your environment's python is used).
4.  **Save & Start**: Click **💾 Save Config**, then click **▶ Start Agent**.
    *   The agent will immediately start its first cycle, analyzing the diabetes data and generating model hypotheses.

## Location
`example/diabetes_sklearn`

## Structure
*   `strategy.py`: Defines the Model and Hyperparameter Space.
*   `evaluator.py`: Runs Grid Search and reports MSE.
