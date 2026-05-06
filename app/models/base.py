from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BaseMLModel(ABC):
    """Базовая обертка над сохраненной sklearn-моделью."""

    def __init__(self, model_path: Path, meta_path: Path) -> None:
        """Инициализирует обертку и загружает модель с метаданными."""

        self.model_path = model_path
        self.meta_path = meta_path
        self._model: Any | None = None
        self._metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Загружает файлы модели и метаданных из файловой системы."""

        if not self.model_path.exists():
            raise FileNotFoundError(f'Файл модели не найден: {self.model_path}')
        if not self.meta_path.exists():
            raise FileNotFoundError(f'Файл метаданных не найден: {self.meta_path}')

        self._model = joblib.load(self.model_path)
        self._metadata = joblib.load(self.meta_path)

    @property
    def is_loaded(self) -> bool:
        """Возвращает True, если модель и метаданные успешно загружены."""

        return self._model is not None and bool(self._metadata)

    @property
    def metadata(self) -> dict[str, Any]:
        """Возвращает исходные метаданные модели."""

        return self._metadata

    @property
    def feature_names(self) -> list[str]:
        """Возвращает список названий входных признаков."""

        return list(self._metadata.get('feature_names', []))

    @property
    def target_names(self) -> list[str]:
        """Возвращает список человекочитаемых названий классов."""

        return list(self._metadata.get('target_names', []))

    @property
    def accuracy(self) -> float:
        """Возвращает accuracy, рассчитанную на отложенной выборке."""

        return float(self._metadata.get('accuracy', 0.0))

    @property
    def metrics(self) -> dict[str, float]:
        """Возвращает числовые метрики, сохраненные при обучении модели."""

        return {
            key: float(value)
            for key, value in self._metadata.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }

    @abstractmethod
    def prepare_features(self, **kwargs: Any) -> np.ndarray:
        """Преобразует поля запроса в двумерный массив признаков для sklearn."""

    def _get_class_name(self, predicted_label: Any, class_position: int) -> str:
        """Возвращает человекочитаемое название класса по метке модели."""

        if 0 <= class_position < len(self.target_names):
            return self.target_names[class_position]

        try:
            label_index = int(predicted_label)
        except (TypeError, ValueError):
            return str(predicted_label)

        if 0 <= label_index < len(self.target_names):
            return self.target_names[label_index]

        return str(predicted_label)

    def predict(self, **kwargs: Any) -> dict[str, Any]:
        """Выполняет предсказание и возвращает класс, вероятность и уверенность."""

        if self._model is None:
            raise RuntimeError('Модель не загружена')

        features = self.prepare_features(**kwargs)
        predicted_label = self._model.predict(features)[0]
        probabilities = self._model.predict_proba(features)[0]
        class_labels = list(getattr(self._model, 'classes_', range(len(probabilities))))

        try:
            class_position = class_labels.index(predicted_label)
        except ValueError:
            class_position = int(np.argmax(probabilities))

        probability_by_class = {
            self._get_class_name(label, index): float(probability)
            for index, (label, probability) in enumerate(zip(class_labels, probabilities))
        }

        return {
            'predicted_class': self._get_class_name(predicted_label, class_position),
            'class_index': int(class_position),
            'probabilities': probability_by_class,
            'confidence': float(probabilities[class_position]),
        }
