import numpy as np
import pandas as pd
import os
import sys

# Create directories for data storage
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)


# Data generation
def generate_data(n_samples,
                  anomaly_ratio=0.1,
                  anomaly_loc=30,
                  anomaly_scale=10):

    # Data generation without anomalies
    data = np.random.normal(loc=20, scale=5, size=(n_samples, 1))

    # Calculating the number of anomalies
    n_anomalies = int(n_samples * anomaly_ratio)

    # Adding anomalies
    anomalies = np.random.normal(loc=anomaly_loc, scale=anomaly_scale,
                                 size=(n_anomalies, 1))
    data = np.concatenate((data, anomalies), axis=0)

    # Rounding data to one decimal place
    data = np.round(data, 1)

    # Creating a second column with anomaly labels
    labels = np.zeros(data.size, dtype=int)
    labels[n_samples:] = 1  # Label 1 for anomalies

    # Creating a structured array (list of tuples)
    dtype = [('data', np.float32), ('labels', np.int32)]
    data_with_labels = np.empty(data.size, dtype=dtype)
    data_with_labels['data'] = data.flatten()
    data_with_labels['labels'] = labels

    # Creating a dictionary from a list of tuples
    data_dict = {'temperature': [temp for temp, anomaly in data_with_labels],
                 'anomaly': [anomaly for temp, anomaly in data_with_labels]}

    return data_dict


# Getting the number of data sets
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Default value if no argument is passed

for i in range(n_datasets):
    # Generating and storing training data
    train_data = generate_data(1000,
                               anomaly_ratio=0.1,
                               anomaly_loc=30+i*5,
                               anomaly_scale=10+i*2)
    df_train = pd.DataFrame(train_data)
    df_train.to_csv(f'train/temperature_train_{i+1}.csv', index=False)

    # Generating and storing testing data
    test_data = generate_data(200,
                              anomaly_ratio=0.1,
                              anomaly_loc=30+i*5,
                              anomaly_scale=10+i*2)
    df_test = pd.DataFrame(test_data)
    df_test.to_csv(f'test/temperature_test_{i+1}.csv', index=False)
