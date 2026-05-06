from fastapi import APIRouter

from app.dependencies import IrisModelDep
from app.schemas.prediction import IrisFeatures, IrisPredictionResponse, ModelInfo


router = APIRouter(prefix='/api/iris', tags=['Iris Classification'])


@router.get('/info', response_model=ModelInfo)
async def get_iris_info(model: IrisModelDep) -> ModelInfo:
    """Возвращает сведения о модели классификации ирисов."""

    return ModelInfo(
        name='Iris Species Classifier',
        description='Классификация сортов ирисов по размерам чашелистиков и лепестков',
        features=model.feature_names,
        classes=model.target_names,
        accuracy=model.accuracy,
        metrics=model.metrics,
    )


@router.post('/predict', response_model=IrisPredictionResponse)
async def predict_iris(features: IrisFeatures, model: IrisModelDep) -> IrisPredictionResponse:
    """Выполняет предсказание сорта ириса."""

    result = model.predict(**features.model_dump())

    return IrisPredictionResponse(
        predicted_class=result['predicted_class'],
        class_index=result['class_index'],
        probabilities=result['probabilities'],
        confidence=result['confidence'],
        features=features,
    )
