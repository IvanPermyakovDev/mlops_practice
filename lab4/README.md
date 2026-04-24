# Lab 4

В этой лабораторной показано версионирование датасета с помощью Git и DVC.

Реализовано:

- инициализация DVC в Git-репозитории;
- DVC remote `gdrive` в папке Google Drive
  `gdrive://1pjuisQNYEow2PQkAOXQ1mEPvNJfET_Rd`;
- версионируемый датасет `lab4/data/titanic.csv`;
- три версии датасета с тегами Git:
  - `lab4-selected` - признаки `Pclass`, `Sex`, `Age`;
  - `lab4-age-filled` - пропуски `Age` заполнены средним возрастом;
  - `lab4-sex-encoded` - добавлены one-hot признаки `Sex_female`, `Sex_male`;
- отправка DVC-объектов в remote через `dvc push`;
- переключение между версиями через `git checkout` и `dvc checkout`.

## Подготовка окружения

```bash
python3 -m venv .venv-lab4
source .venv-lab4/bin/activate
pip install -r lab4/requirements.txt
```

## Проверка текущего датасета

```bash
source .venv-lab4/bin/activate
python lab4/src/inspect_dataset.py lab4/data/titanic.csv
dvc status
```

## Переключение между версиями

```bash
source .venv-lab4/bin/activate
bash lab4/switch_version.sh lab4-selected
bash lab4/switch_version.sh lab4-age-filled
bash lab4/switch_version.sh lab4-sex-encoded
```

После переключения команда печатает форму датасета, список колонок и количество
пропусков в `Age`.

## Воспроизведение отдельных версий

```bash
python lab4/src/build_dataset.py --version selected --output lab4/data/titanic.csv
python lab4/src/build_dataset.py --version age_filled --output lab4/data/titanic.csv
python lab4/src/build_dataset.py --version sex_encoded --output lab4/data/titanic.csv
```

Скрипт загружает Titanic через `catboost.datasets.titanic()`

## Команды DVC, использованные в работе

```bash
dvc init
dvc remote add -d gdrive gdrive://1pjuisQNYEow2PQkAOXQ1mEPvNJfET_Rd
python lab4/src/build_dataset.py --version selected --output lab4/data/titanic.csv
dvc add lab4/data/titanic.csv
dvc push

python lab4/src/build_dataset.py --version age_filled --output lab4/data/titanic.csv
dvc add lab4/data/titanic.csv
dvc push

python lab4/src/build_dataset.py --version sex_encoded --output lab4/data/titanic.csv
dvc add lab4/data/titanic.csv
dvc push
```

## Примечание по Google Drive OAuth

Удаленное хранилище настроено на папку Google Drive:

```bash
dvc remote list
# gdrive  gdrive://1pjuisQNYEow2PQkAOXQ1mEPvNJfET_Rd  (default)
```