from functools import lru_cache
from pathlib import Path


class Settings:
    """Настройки приложения с путями к ресурсам проекта."""

    BASE_DIR = Path(__file__).parent.parent

    MODELS_DIR = BASE_DIR / 'app' / 'models'
    STATIC_DIR = BASE_DIR / 'static'
    TEMPLATES_DIR = BASE_DIR / 'templates'

    IRIS_MODEL_PATH = MODELS_DIR / 'iris_model.pkl'
    IRIS_META_PATH = MODELS_DIR / 'iris_model.meta.pkl'

    WINE_MODEL_PATH = MODELS_DIR / 'wine_model.pkl'
    WINE_META_PATH = MODELS_DIR / 'wine_model.meta.pkl'

    TITANIC_MODEL_PATH = MODELS_DIR / 'titanic_model.pkl'
    TITANIC_META_PATH = MODELS_DIR / 'titanic_model.meta.pkl'


@lru_cache
def get_settings() -> Settings:
    """Возвращает кэшированный экземпляр настроек приложения."""

    return Settings()
