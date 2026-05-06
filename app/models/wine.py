from pathlib import Path

import numpy as np

from app.models.base import BaseMLModel


class WineModel(BaseMLModel):
    """Модель классификации вина по физико-химическим признакам."""

    def __init__(self, model_path: Path, meta_path: Path) -> None:
        """Создает обертку для модели Wine."""

        super().__init__(model_path, meta_path)

    def prepare_features(
        self,
        alcohol: float,
        malic_acid: float,
        ash: float,
        alcalinity_of_ash: float,
        magnesium: float,
        total_phenols: float,
        flavanoids: float,
        nonflavanoid_phenols: float,
        proanthocyanins: float,
        color_intensity: float,
        hue: float,
        od280_od315: float,
        proline: float,
    ) -> np.ndarray:
        """Формирует массив признаков для модели Wine."""

        return np.array([[
            alcohol,
            malic_acid,
            ash,
            alcalinity_of_ash,
            magnesium,
            total_phenols,
            flavanoids,
            nonflavanoid_phenols,
            proanthocyanins,
            color_intensity,
            hue,
            od280_od315,
            proline,
        ]])
