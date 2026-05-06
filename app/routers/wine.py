from fastapi import APIRouter

from app.dependencies import WineModelDep
from app.schemas.prediction import (
    ModelInfo,
    WineFeatures,
    WinePredictionResponse,
)

router = APIRouter(prefix='/api/wine', tags=['Wine Classification'])


CLASS_DESCRIPTIONS = {
    'Культивар 1': 'Вино сорта 0 — культивар 1',
    'Культивар 2': 'Вино сорта 1 — культивар 2',
    'Культивар 3': 'Вино сорта 2 — культивар 3',
}


@router.get('/info', response_model=ModelInfo)
async def get_wine_info(model: WineModelDep) -> ModelInfo:
    """Возвращает сведения о модели классификации вина."""

    return ModelInfo(
        name='Wine Classifier',
        description='Классификация сортов вина по физико-химическим показателям',
        features=model.feature_names,
        classes=model.target_names,
        accuracy=model.accuracy,
        metrics=model.metrics,
    )


@router.post('/predict', response_model=WinePredictionResponse)
async def predict_wine(
    features: WineFeatures, model: WineModelDep
) -> WinePredictionResponse:
    """Выполняет предсказание сорта вина."""

    result = model.predict(**features.model_dump())
    predicted_class = result['predicted_class']

    return WinePredictionResponse(
        predicted_class=predicted_class,
        class_index=result['class_index'],
        probabilities=result['probabilities'],
        confidence=result['confidence'],
        features=features,
        description=CLASS_DESCRIPTIONS.get(
            predicted_class, 'Описание класса отсутствует'
        ),
    )
