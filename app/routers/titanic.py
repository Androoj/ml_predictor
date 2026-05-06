from fastapi import APIRouter

from app.dependencies import TitanicModelDep
from app.schemas.prediction import ModelInfo, TitanicFeatures, TitanicPredictionResponse


router = APIRouter(prefix='/api/titanic', tags=['Titanic Survival'])


@router.get('/info', response_model=ModelInfo)
async def get_titanic_info(model: TitanicModelDep) -> ModelInfo:
    """Возвращает сведения о модели предсказания выживания на Титанике."""

    return ModelInfo(
        name='Titanic Survival Predictor',
        description='Предсказание вероятности выживания пассажира Титаника',
        features=model.feature_names,
        classes=model.target_names,
        accuracy=model.accuracy,
        metrics=model.metrics,
    )


@router.post('/predict', response_model=TitanicPredictionResponse)
async def predict_survival(
    features: TitanicFeatures,
    model: TitanicModelDep,
) -> TitanicPredictionResponse:
    """Выполняет предсказание выживания пассажира Титаника."""

    result = model.predict(**features.model_dump())
    survival_percentage = result['probabilities'].get('Survived', 0.0) * 100

    return TitanicPredictionResponse(
        predicted_class=result['predicted_class'],
        class_index=result['class_index'],
        probabilities=result['probabilities'],
        confidence=result['confidence'],
        features=features,
        survival_percentage=survival_percentage,
    )
