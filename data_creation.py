import numpy as np
import pandas as pd
import os
import sys

# Создаем директории для хранения данных
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)


# Генерация данных
def generate_data(n_samples,
                  anomaly_ratio=0.1,
                  anomaly_loc=30,
                  anomaly_scale=10):
    # Генерация данных без аномалий
    data = np.random.normal(loc=20, scale=5, size=(n_samples, 1))
    # Добавление аномалий
    anomalies = np.random.normal(loc=anomaly_loc, scale=anomaly_scale,
                                 size=(int(n_samples * anomaly_ratio), 1))
    data = np.concatenate((data, anomalies), axis=0)
    # Округление данных до одного десятичного знака
    data = np.round(data, 1)
    return data


# Получение количества наборов данных
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Значение по умолчанию, если аргумент не передан

for i in range(n_datasets):
    # Генерация и сохранение данных для обучения
    train_data = generate_data(1000,
                               anomaly_ratio=0.1,
                               anomaly_loc=30+i*5,
                               anomaly_scale=10+i*2)
    df_train = pd.DataFrame(train_data, columns=['temperature'])
    df_train.to_csv(f'train/temperature_train_{i+1}.csv', index=False)

    # Генерация и сохранение данных для тестирования
    test_data = generate_data(200,
                              anomaly_ratio=0.1,
                              anomaly_loc=30+i*5,
                              anomaly_scale=10+i*2)
    df_test = pd.DataFrame(test_data, columns=['temperature'])
    df_test.to_csv(f'test/temperature_test_{i+1}.csv', index=False)
