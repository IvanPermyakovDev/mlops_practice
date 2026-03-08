FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY lab3/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY lab3 ./lab3

RUN python -m lab3.train_model

EXPOSE 8000

CMD ["uvicorn", "lab3.app:app", "--host", "0.0.0.0", "--port", "8000"]
