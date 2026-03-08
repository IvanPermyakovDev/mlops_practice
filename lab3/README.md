# Lab 3

В этой лабораторной реализован микросервис с моделью машинного обучения в Docker.

Содержимое:

- `train_model.py` - обучение модели Iris и сохранение в `pickle`
- `app.py` - FastAPI микросервис с endpoint `POST /predict`
- `Dockerfile` - сборка образа
- `docker-compose.yml` - запуск через compose

Пример локального запуска:

```bash
docker build -t lab3-iris-service -f lab3/Dockerfile .
docker run --rm -p 8000:8000 lab3-iris-service
```

Проверка:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
