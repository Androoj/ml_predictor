from pathlib import Path
from typing import Any

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'app' / 'models'


def save_model_artifacts(model: Any, metadata: dict[str, Any], model_name: str) -> Path:
    """Сохраняет модель и метаданные в стандартную директорию проекта."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f'{model_name}_model.pkl'
    meta_path = MODELS_DIR / f'{model_name}_model.meta.pkl'

    joblib.dump(model, model_path)
    joblib.dump(metadata, meta_path)

    return MODELS_DIR
