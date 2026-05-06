from pathlib import Path

import numpy as np

from app.models.base import BaseMLModel


class IrisModel(BaseMLModel):
    """Модель классификации ирисов по размерам чашелистика и лепестка."""

    def __init__(self, model_path: Path, meta_path: Path) -> None:
        """Создает обертку для модели Iris."""

        super().__init__(model_path, meta_path)

    def prepare_features(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> np.ndarray:
        """Формирует массив признаков для модели Iris."""

        return np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        ]])
