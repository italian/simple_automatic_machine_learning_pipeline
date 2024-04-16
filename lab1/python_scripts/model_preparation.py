import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import os
import sys


# Function for model training and metrics calculation
def train_model_and_evaluate(file_path):
    # Data loading
    df = pd.read_csv(file_path)

    # Data shuffling
    # df = df.sample(frac=1).reset_index(drop=True)
    df = shuffle(df, random_state=42)

    # Separating data into features and target variable
    X = df[['temperature']]  # type: ignore
    y = df['anomaly']  # type: ignore

    # Creating an instance of the logistic regression model
    model = LogisticRegression()

    # Training of the model
    model.fit(X, y)

    # Prediction on training data
    y_pred = model.predict(X)

    # Calculation of metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Creating a DataFrame for the results
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return model, results


# Getting the number of data sets
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Default value if no argument is passed

# Creating a directory for storing models
os.makedirs('models', exist_ok=True)
print()
for i in range(n_datasets):
    # Training the model on preprocessed data
    model, results = train_model_and_evaluate(
        f'train/temperature_train_{i+1}_preprocessed.csv')

    # Saving the trained model
    joblib.dump(model, f'models/model_{i+1}.pkl')

    print(f"The model for the data set {i+1} is trained.")
    print(results.to_string(index=False))
    print('-' * 20)
