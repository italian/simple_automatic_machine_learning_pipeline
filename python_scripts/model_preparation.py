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


# Функция для обучения модели и вычисления метрик
def train_model_and_evaluate(file_path):
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Перемешивание данных
    # df = df.sample(frac=1).reset_index(drop=True)
    df = shuffle(df, random_state=42)

    # Разделение данных на признаки и целевую переменную
    X = df[['temperature']]  # type: ignore
    y = df['anomaly']  # type: ignore

    # Создание экземпляра модели логистической регрессии
    model = LogisticRegression()

    # Обучение модели
    model.fit(X, y)

    # Предсказание на обучающих данных
    y_pred = model.predict(X)

    # Вычисление метрик
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Создание DataFrame для результатов
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return model, results


# Получение количества наборов данных
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Значение по умолчанию, если аргумент не передан

# Создание директории для хранения моделей
os.makedirs('models', exist_ok=True)

for i in range(n_datasets):
    # Обучение модели на предобработанных данных
    model, results = train_model_and_evaluate(
        f'train/temperature_train_{i+1}_preprocessed.csv')

    # Сохранение обученной модели
    joblib.dump(model, f'models/model_{i+1}.pkl')

    print(f"Модель для набора данных {i+1} обучена.")
    print(results.to_string(index=False))
    print('-' * 20)
