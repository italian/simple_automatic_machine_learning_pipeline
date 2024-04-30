[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/italian/simple_automatic_machine_learning_pipeline/blob/main/LICENSE)

# Простая автоматизированная система машинного обучения

Это реализация простой автоматизированной системы машинного обучения, использующей [Docker](https://www.docker.com) и [Streamlit](https://streamlit.io) для развертывания модели логистической регрессии на датасете Iris.

## Как начать

1. Клонируйте репозиторий.
2. Перейдите в директорию проекта:
    ```shell
    cd simple_automatic_machine_learning_pipeline/lab3
    ```
3. Запустите Docker Compose:
    ```shell
    docker-compose up
    ```
4. Откройте в браузере `http://localhost:8501` для доступа к приложению Streamlit.

## Как использовать

Приложение Streamlit позволяет вводить параметры для предсказания вида ириса.

Используйте слайдеры для ввода длины и ширины чашелистика и лепестка, а затем нажмите кнопку "Predict" для получения предсказания модели.

![Скриншот приложения](./screenshots/app.png)

## Лицензия

Этот проект распространяется под лицензией MIT.