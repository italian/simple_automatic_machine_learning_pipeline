import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler


# Function for data preprocessing
def preprocess_data(file_path):
    # Data loading
    df = pd.read_csv(file_path)

    # Creating a StandardScaler instance
    scaler = StandardScaler()

    # Applying StandardScaler to data
    scaled_data = scaler.fit_transform(df[['temperature']])

    # Saving preprocessed data
    df['temperature'] = scaled_data

    return df


# Getting the number of data sets
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Default value if no argument is passed

for i in range(n_datasets):
    # Preprocessing and storing data for training
    train_data_preprocessed = preprocess_data(
        f'train/temperature_train_{i+1}.csv')
    train_data_preprocessed.to_csv(
        f'train/temperature_train_{i+1}_preprocessed.csv', index=False)

    # Preprocessing and saving data for testing
    test_data_preprocessed = preprocess_data(
        f'test/temperature_test_{i+1}.csv')
    test_data_preprocessed.to_csv(
        f'test/temperature_test_{i+1}_preprocessed.csv', index=False)
