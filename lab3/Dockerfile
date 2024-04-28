FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY python_scripts/model.py .

CMD ["streamlit", "run", "model.py"]
