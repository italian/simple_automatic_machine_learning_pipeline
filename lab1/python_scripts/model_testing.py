import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import sys


# Function for model testing
def test_model(model_path, test_data_path):
    # Loading a trained model
    model = joblib.load(model_path)

    # Loading test data
    df_test = pd.read_csv(test_data_path)

    # Separating data into features and target variable
    X_test = df_test[['temperature']]
    y_test = df_test['anomaly']

    # Prediction on test data
    y_pred = model.predict(X_test)

    # Calculation of metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Creating a DataFrame for the results
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return results


# Getting the number of datasets
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Default value if no argument is passed
print()
for i in range(n_datasets):
    # Path to a trained model
    model_path = f'models/model_{i+1}.pkl'
    # Path to test data
    test_data_path = f'test/temperature_test_{i+1}_preprocessed.csv'

    # Model testing
    results = test_model(model_path, test_data_path)
    print(f"The model for dataset {i+1} is tested.")
    print(results.to_string(index=False))
    print('-' * 20)
