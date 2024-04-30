import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler


# Function for data preprocessing
def preprocess_data(train_file_path, test_file_path):
    # Train data loading
    train_df = pd.read_csv(train_file_path)
    # Test data loading
    test_df = pd.read_csv(test_file_path)

    # Creating a StandardScaler instance
    scaler = StandardScaler()

    # Training StandardScaler on training data
    scaler.fit(train_df[['temperature']])

    # Applying StandardScaler to train data
    train_scaled_data = scaler.transform(train_df[['temperature']])
    # Applying StandardScaler to test data
    test_scaled_data = scaler.transform(test_df[['temperature']])

    # Saving scaled training data
    train_df['temperature'] = train_scaled_data
    train_df.to_csv(
        train_file_path.replace('.csv', '_preprocessed.csv'), index=False)

    # Saving scaled test data
    test_df['temperature'] = test_scaled_data
    test_df.to_csv(
        test_file_path.replace('.csv', '_preprocessed.csv'), index=False)


# Getting the number of data sets
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Default value if no argument is passed

for i in range(n_datasets):
    # Preprocessing and storing data for training and testing
    preprocess_data(
        f'train/temperature_train_{i+1}.csv',
        f'test/temperature_test_{i+1}.csv')
