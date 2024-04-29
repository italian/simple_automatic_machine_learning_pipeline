from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np

# Загрузка датасета
iris = load_iris()
X = iris.data  # type: ignore
y = iris.target  # type: ignore

# Разделение датасета на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)

# Создание и обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Функция для предсказания
def predict(sample):
    prediction = model.predict(sample)
    return prediction


# Веб-интерфейс с помощью Streamlit
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
    prediction = predict(sample)
    st.write(
        f'Предсказание: {iris.target_names[prediction[0]]}')  # type: ignore
