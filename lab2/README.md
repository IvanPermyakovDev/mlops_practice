# Lab 2

Содержимое каталога:

- `data_collection.py` - загрузка набора данных Iris из UCI и разбиение на train/test.
- `data_preprocessing.py` - масштабирование признаков и кодирование целевой переменной.
- `model_training.py` - обучение модели `LogisticRegression` и сохранение в `pickle`.
- `model_evaluation.py` - загрузка модели и оценка качества на тестовой выборке.
- `run_pipeline.sh` - локальный запуск всех этапов.
- `jenkins_pipeline` - Jenkins declarative pipeline для CI/CD сценария.

Локальный запуск:

```bash
bash lab2/run_pipeline.sh
```
