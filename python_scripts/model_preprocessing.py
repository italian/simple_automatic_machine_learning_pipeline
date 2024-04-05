import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler


# Функция для предобработки данных
def preprocess_data(file_path):
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Создание экземпляра StandardScaler
    scaler = StandardScaler()

    # Применение StandardScaler к данным
    scaled_data = scaler.fit_transform(df[['temperature']])

    # Сохранение предобработанных данных
    df['temperature'] = scaled_data

    return df


# Получение количества наборов данных
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Значение по умолчанию, если аргумент не передан

for i in range(n_datasets):
    # Предобработка и сохранение данных для обучения
    train_data_preprocessed = preprocess_data(
        f'train/temperature_train_{i+1}.csv')
    train_data_preprocessed.to_csv(
        f'train/temperature_train_{i+1}_preprocessed.csv', index=False)

    # Предобработка и сохранение данных для тестирования
    test_data_preprocessed = preprocess_data(
        f'test/temperature_test_{i+1}.csv')
    test_data_preprocessed.to_csv(
        f'test/temperature_test_{i+1}_preprocessed.csv', index=False)
