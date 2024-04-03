#!/bin/bash

# Функция для создания виртуальной среды
create_venv() {
    local env_name=${1:-".venv"}
    python3 -m venv "$env_name"
    echo "Виртуальная среда '$env_name' создана."
}

# Функция для активации виртуальной среды
activate_venv() {
    local env_name=${1:-".venv"}
    if [ ! -d "$env_name" ]; then
        echo "Виртуальная среда '$env_name' не найдена. Используйте '$0 create [env_name]' для создания."
        return 1
    fi
    if [ -z "$VIRTUAL_ENV" ]; then
        source "./$env_name/bin/activate"
        echo "Виртуальная среда '$env_name' активирована."
    else
        echo "Виртуальная среда уже активирована."
    fi
}

# Функция для установки зависимостей из requirements.txt
install_deps() {
    if [ ! -f "requirements.txt" ]; then
        echo "Файл requirements.txt не найден."
        return 1
    fi

    # Проверка, установлены ли все зависимости из requirements.txt
    for package in $(cat requirements.txt | cut -d '=' -f 1); do
        if ! pip freeze | grep -q "^$package=="; then
            echo "Установка зависимостей..."
            pip install -r requirements.txt
            echo "Зависимости установлены."
            return 0
        fi
    done

    echo "Все зависимости уже установлены."
}

# Создание виртуальной среды, если она еще не создана
if [ ! -d ".venv" ]; then
    create_venv
fi

# Активация виртуальной среды
activate_venv

# Установка зависимостей
install_deps

# Получение количества наборов данных
n_datasets=$1

# Запуск скрипта создания данных
python data_creation.py $n_datasets

# Запуск скрипта предобработки данных
python model_preprocessing.py

# Запуск скрипта подготовки и обучения модели
python model_preparation.py

# Запуск скрипта тестирования модели
python model_testing.py
