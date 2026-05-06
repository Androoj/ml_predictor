from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


ClassName = Annotated[str, StringConstraints(min_length=1, max_length=100)]
Description = Annotated[str, StringConstraints(min_length=1, max_length=500)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class ApiMessage(BaseModel):
    """Унифицированное сообщение API для ошибок и служебных ответов."""

    message: str
    detail: str | list | dict | None = None


class IrisFeatures(BaseModel):
    """Входные признаки для классификации ириса."""

    model_config = ConfigDict(extra='forbid')

    sepal_length: Annotated[float, Field(ge=4.0, le=8.0, description='Длина чашелистика')]
    sepal_width: Annotated[float, Field(ge=2.0, le=4.5, description='Ширина чашелистика')]
    petal_length: Annotated[float, Field(ge=1.0, le=7.0, description='Длина лепестка')]
    petal_width: Annotated[float, Field(ge=0.1, le=2.5, description='Ширина лепестка')]


class WineFeatures(BaseModel):
    """Входные признаки для классификации вина."""

    model_config = ConfigDict(extra='forbid')

    alcohol: Annotated[float, Field(ge=11.0, le=15.0, description='Содержание алкоголя')]
    malic_acid: Annotated[float, Field(ge=0.5, le=6.0, description='Яблочная кислота')]
    ash: Annotated[float, Field(ge=1.0, le=3.5, description='Зола')]
    alcalinity_of_ash: Annotated[float, Field(ge=10.0, le=30.0, description='Щелочность золы')]
    magnesium: Annotated[float, Field(ge=50.0, le=170.0, description='Магний')]
    total_phenols: Annotated[float, Field(ge=0.5, le=4.0, description='Всего фенолов')]
    flavanoids: Annotated[float, Field(ge=0.1, le=5.0, description='Флавоноиды')]
    nonflavanoid_phenols: Annotated[float, Field(ge=0.1, le=1.0, description='Нефлавоноидные фенолы')]
    proanthocyanins: Annotated[float, Field(ge=0.1, le=4.0, description='Проантоцианидины')]
    color_intensity: Annotated[float, Field(ge=1.0, le=13.0, description='Интенсивность цвета')]
    hue: Annotated[float, Field(ge=0.4, le=1.8, description='Оттенок')]
    od280_od315: Annotated[float, Field(ge=1.0, le=4.0, description='OD280/OD315 разбавленных вин')]
    proline: Annotated[float, Field(ge=200.0, le=1700.0, description='Пролин')]


class TitanicFeatures(BaseModel):
    """Входные признаки для предсказания выживания на Титанике."""

    model_config = ConfigDict(extra='forbid')

    pclass: Annotated[int, Field(ge=1, le=3, description='Класс билета')]
    sex: Annotated[Literal[0, 1], Field(description='Пол: 0 — женщина, 1 — мужчина')]
    age: Annotated[float, Field(ge=0.0, le=100.0, description='Возраст')]
    sibsp: Annotated[int, Field(ge=0, le=8, description='Братья, сестры или супруг на борту')]
    parch: Annotated[int, Field(ge=0, le=6, description='Родители или дети на борту')]
    fare: Annotated[float, Field(ge=0.0, le=600.0, description='Стоимость билета')]
    embarked: Annotated[Literal[0, 1, 2], Field(description='Порт посадки: 0 — S, 1 — C, 2 — Q')]


class PredictionResponse(BaseModel):
    """Базовая схема ответа с результатом предсказания."""

    predicted_class: ClassName
    class_index: int
    probabilities: dict[ClassName, Probability]
    confidence: Probability


class IrisPredictionResponse(PredictionResponse):
    """Ответ API для модели Iris."""

    features: IrisFeatures


class WinePredictionResponse(PredictionResponse):
    """Ответ API для модели Wine."""

    features: WineFeatures
    description: str = ''


class TitanicPredictionResponse(PredictionResponse):
    """Ответ API для модели Titanic."""

    features: TitanicFeatures
    survival_percentage: Annotated[float, Field(ge=0.0, le=100.0)]


class ModelInfo(BaseModel):
    """Информация о подключенной ML-модели."""

    name: ClassName
    description: Description
    features: list[ClassName]
    classes: list[ClassName]
    accuracy: Probability
    metrics: dict[str, float] = Field(default_factory=dict)
