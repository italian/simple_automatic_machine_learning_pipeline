# Используйте официальный образ Python
FROM python:3.8-slim

# Установите рабочую директорию в контейнере
WORKDIR /streamlit_app

# Копируйте файлы
COPY . .

# Установите зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем приложение
CMD ["streamlit", "run", "streamlit_app.py"]
