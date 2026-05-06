# ML Predictor

ML Predictor — небольшое веб-приложение на **FastAPI** для демонстрации трех задач машинного обучения:

- классификация ирисов по размерам чашелистика и лепестка;
- классификация вина по физико-химическим признакам;
- предсказание выживания пассажира Титаника.

Проект содержит REST API, HTML/CSS/JavaScript-интерфейс и отдельные скрипты обучения моделей на scikit-learn.

## Стек

- Python 3.12+
- FastAPI
- Pydantic
- Uvicorn
- scikit-learn
- pandas
- NumPy
- Joblib
- HTML/CSS/JavaScript

## Структура проекта

```text
ml_predictor/
├── app/
│   ├── models/              # Обертки над ML-моделями
│   ├── routers/             # API-роутеры
│   ├── schemas/             # Pydantic-схемы запросов и ответов
│   ├── config.py            # Пути к директориям и моделям
│   ├── dependencies.py      # FastAPI-зависимости и кэширование моделей
│   └── main.py              # Создание FastAPI-приложения
├── ml/
│   ├── common.py            # Общая логика сохранения артефактов модели
│   ├── train_iris.py        # Обучение модели Iris
│   ├── train_wine.py        # Обучение модели Wine
│   └── train_titanic.py     # Обучение модели Titanic
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
├── requirements.txt
├── run.py
└── README.md
```

## Быстрый старт

Проект использует `uv` для управления зависимостями и виртуальным окружением.

### 1. Установка uv

Если `uv` еще не установлен, его можно установить через `pip`:

```bash
pip install uv
```

Если необходимо установить uv в конкретную версиюю Python

```bash
py -ВЕРСИЯ_PYTHON -m pip install uv
```

### 2. Установить зависимости

Команда создаст виртуальное окружение .venv, установит зависимости из pyproject.toml и создаст/обновит uv.lock.

```bash
uv sync
```

### 3. Обучить модели

Файлы моделей не хранятся в репозитории, потому что `.pkl` и `.joblib` исключены через `.gitignore`. Перед запуском предсказаний нужно выполнить:

```bash
uv run python ml/train_iris.py
uv run python ml/train_wine.py
uv run python ml/train_titanic.py
```

После выполнения команд в app/models/ появятся файлы моделей:

```text
iris_model.pkl
iris_model.meta.pkl
wine_model.pkl
wine_model.meta.pkl
titanic_model.pkl
titanic_model.meta.pkl
```

> Скрипт `ml/train_titanic.py` загружает датасет Titanic из публичного CSV по URL. Для его выполнения нужен доступ в интернет.

### 4. Запустить приложение

```bash
uv run run.py
```

Или напрямую через Uvicorn:

```bash
uv run uvicorn app.main:app --reload
```

По умолчанию приложение будет доступно на:

```text
http://localhost:8000
```

## API

### Web UI

```text
GET /
```

### Healthcheck

```text
GET /health
```

### Iris

```text
GET  /api/iris/info
POST /api/iris/predict
```

### Wine

```text
GET  /api/wine/info
POST /api/wine/predict
```

### Titanic

```text
GET  /api/titanic/info
POST /api/titanic/predict
```

### Документация API

```text
Swagger UI: http://localhost:8000/api/docs
ReDoc:      http://localhost:8000/api/redoc
```

## Что было улучшено

- Исправлены зависимости `fastapi[standard]` и `uvicorn[standard]`.
- Добавлены корректные `__init__.py` для пакетов.
- Модели кэшируются и не загружаются заново при каждом запросе.
- Ошибка отсутствующих `.pkl`-файлов возвращается как понятный `503 Service Unavailable`.
- Pydantic-схемы стали строже: лишние поля запрещены, диапазоны значений заданы в схемах.
- Ответы `/info` дополнены метриками модели.
- Добавлен единый формат ошибок валидации.
- Убран прямой доступ к приватным полям моделей из роутеров.
- ML-скрипты обучения используют общую функцию сохранения артефактов.
- Frontend убрал дублирование сбора полей и показывает ошибки API на странице.
- Все добавленные docstring-и написаны на русском языке.

## Примечания

Тесты в проект не добавлялись по отдельному требованию.
