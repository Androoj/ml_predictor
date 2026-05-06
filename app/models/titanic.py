from pathlib import Path

import numpy as np

from app.models.base import BaseMLModel


class TitanicModel(BaseMLModel):
    """Модель предсказания выживания пассажира Титаника."""

    def __init__(self, model_path: Path, meta_path: Path) -> None:
        """Создает обертку для модели Titanic."""

        super().__init__(model_path, meta_path)

    def prepare_features(
        self,
        pclass: int,
        sex: int,
        age: float,
        sibsp: int,
        parch: int,
        fare: float,
        embarked: int,
    ) -> np.ndarray:
        """Формирует массив признаков для модели Titanic."""

        return np.array([[
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare,
            embarked,
        ]])
