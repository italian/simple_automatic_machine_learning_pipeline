[![ru](https://img.shields.io/badge/lang-ru-red.svg)](https://github.com/italian/simple_automatic_machine_learning_pipeline/blob/main/README.ru.md)

# Simple automatic machine learning pipeline

The repository is an educational project aimed at creating a simple pipeline for automating work with a machine learning model.

The project was developed as part of a training task and includes several key stages.

## Stages

### 1. Data Creation

The script [data_creation.py](python_scripts/data_creation.py) generates various temperature data sets, including anomalies and noise, and stores them in the `train` and `test` folders. Data is generated using the `numpy` library and saved in `CSV` format.

### 2. Data preprocessing

The [model_preprocessing.py](python_scripts/model_preprocessing.py) script performs data standardisation using `sklearn.preprocessing.StandardScaler`. This is necessary to ensure stability of the model learning.

### 3. Обучение модели

Скрипт [model_preparation.py](python_scripts/model_preparation.py) создает и обучает модель машинного обучения на данных из папки `train`. В качестве модели используется логистическая регрессия из библиотеки `sklearn`. После обучения модель сохраняется в формате `.pkl` для дальнейшего использования.

### 4. Тестирование модели

Скрипт [model_testing.py](python_scripts/model_testing.py) проверяет производительность обученной модели на данных из папки `test`. Для оценки производительности используются метрики `accuracy`, `precision`, `recall`, `F1-score`.

### 5. Автоматизация конвейера

Bash-скрипт [pipeline.sh](./pipeline.sh) позволяет запустить весь конвейер автоматически, последовательно запуская все python-скрипты. Скрипт также создает виртуальную среду и устанавливает необходимые зависимости из файла `requirements.txt`.

## Использование

Для запуска конвейера, клонируйте репозиторий и выполните команду `./pipeline.sh` в терминале.

Параметры:

- **Опциональный параметр**: Количество наборов данных для создания. По умолчанию создается один набор данных.

Примеры использования:

- Запуск скрипта без параметров (создается один набор данных):
    ```shell
    ./pipeline.sh
    ```
- Запуск скрипта с указанием количества наборов данных (например, 5 наборов):
    ```shell
    ./pipeline.sh 5
    ```

Убедитесь, что у вас установлены все необходимые зависимости, указанные в файле [requirements.txt](./requirements.txt).

## Требования

- Python 3.7+
- Библиотеки:
    - numpy,
    - pandas,
    - sklearn

## Возможные проблемы и способы их решения

- **Проблема с активацией виртуальной среды в `pipeline.sh`**: Использование в скрипте [pipeline.sh](./pipeline.sh) внутри функции `activate_venv` переменной `$VIRTUAL_ENV` для проверки активации виртуальной среды может быть ненадежным в некоторых случаях, например, когда скрипт запускается внутри сессии `tmux` или `screen`, созданной в активированной виртуальной среде. В таких случаях `$VIRTUAL_ENV` может оставаться установленным, даже если виртуальная среда не активна в текущем контексте.
    - **Возможное решение**: Использовать другие методы для проверки активации виртуальной среды или избегать использования виртуальных сред в таких сценариях.

## Улучшения

- **Добавление поддержки других моделей машинного обучения**: В текущей реализации используется только логистическая регрессия. Добавление поддержки других моделей, таких как деревья решений или нейронные сети, может расширить возможности проекта.
- **Интеграция с системами CI/CD**: Интеграция с системами непрерывной интеграции и непрерывной доставки (CI/CD) может автоматизировать процесс тестирования и развертывания модели, улучшая процесс разработки.

## Лицензия

Этот проект распространяется под лицензией MIT.