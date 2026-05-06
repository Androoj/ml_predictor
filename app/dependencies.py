from functools import lru_cache
from typing import Annotated, TypeVar

from fastapi import Depends, HTTPException, status

from app.config import Settings, get_settings
from app.models.base import BaseMLModel
from app.models.iris import IrisModel
from app.models.titanic import TitanicModel
from app.models.wine import WineModel


SettingsDep = Annotated[Settings, Depends(get_settings)]
ModelT = TypeVar('ModelT', bound=BaseMLModel)


@lru_cache
def load_iris_model() -> IrisModel:
    """Загружает и кэширует модель Iris."""

    settings = get_settings()
    return IrisModel(settings.IRIS_MODEL_PATH, settings.IRIS_META_PATH)


@lru_cache
def load_wine_model() -> WineModel:
    """Загружает и кэширует модель Wine."""

    settings = get_settings()
    return WineModel(settings.WINE_MODEL_PATH, settings.WINE_META_PATH)


@lru_cache
def load_titanic_model() -> TitanicModel:
    """Загружает и кэширует модель Titanic."""

    settings = get_settings()
    return TitanicModel(settings.TITANIC_MODEL_PATH, settings.TITANIC_META_PATH)


def _model_loading_error(exc: FileNotFoundError) -> HTTPException:
    """Формирует понятную ошибку, если файлы модели отсутствуют."""

    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            'message': 'ML-модель недоступна',
            'detail': str(exc),
            'how_to_fix': [
                'python ml/train_iris.py',
                'python ml/train_wine.py',
                'python ml/train_titanic.py',
            ],
        },
    )


def _ensure_loaded(model: ModelT) -> ModelT:
    """Проверяет, что модель загружена, и возвращает ее."""

    if not model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='ML-модель не загружена',
        )
    return model


def get_iris_model(_: SettingsDep) -> IrisModel:
    """Возвращает кэшированную модель Iris для FastAPI-зависимостей."""

    try:
        return _ensure_loaded(load_iris_model())
    except FileNotFoundError as exc:
        raise _model_loading_error(exc) from exc


def get_wine_model(_: SettingsDep) -> WineModel:
    """Возвращает кэшированную модель Wine для FastAPI-зависимостей."""

    try:
        return _ensure_loaded(load_wine_model())
    except FileNotFoundError as exc:
        raise _model_loading_error(exc) from exc


def get_titanic_model(_: SettingsDep) -> TitanicModel:
    """Возвращает кэшированную модель Titanic для FastAPI-зависимостей."""

    try:
        return _ensure_loaded(load_titanic_model())
    except FileNotFoundError as exc:
        raise _model_loading_error(exc) from exc


IrisModelDep = Annotated[IrisModel, Depends(get_iris_model)]
WineModelDep = Annotated[WineModel, Depends(get_wine_model)]
TitanicModelDep = Annotated[TitanicModel, Depends(get_titanic_model)]
