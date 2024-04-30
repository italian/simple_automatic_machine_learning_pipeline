import streamlit as st
import numpy as np
from joblib import load
import pickle

# Загрузка модели и преобразователя
model = load('/model/model.joblib')
with open('/model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Загрузка target_names из файла
with open('/model/target_names.pkl', 'rb') as f:
    target_names = pickle.load(f)

st.title('Модель логистической регрессии на датасете Iris')

# Ввод данных для предсказания
sepal_length = st.slider(
    'Длина чашелистика', min_value=0.0, max_value=10.0, value=0.0,
    step=0.1, format="%.1f")
sepal_width = st.slider(
    'Ширина чашелистика', min_value=0.0, max_value=10.0, value=0.0,
    step=0.1, format="%.1f")
petal_length = st.slider(
    'Длина лепестка', min_value=0.0, max_value=10.0, value=0.0,
    step=0.1, format="%.1f")
petal_width = st.slider(
    'Ширина лепестка', min_value=0.0, max_value=10.0, value=0.0,
    step=0.1, format="%.1f")

if st.button('Предсказать'):
    sample = np.array(
        [sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    # Нормализация входных данных
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    st.write(
        f'Предсказание: {target_names[prediction[0]]}')  # type: ignore
