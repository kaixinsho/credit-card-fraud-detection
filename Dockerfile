FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 FRAUD_DB_PATH=/app/data/predictions.db
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api ./api
COPY app ./app
COPY artifacts ./artifacts
RUN mkdir -p /app/data
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
