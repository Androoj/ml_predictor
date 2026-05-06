import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import Any

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from ml.common import save_model_artifacts

RANDOM_STATE = 42


def train_and_save_wine_model() -> tuple[
    RandomForestClassifier, dict[str, Any]
]:
    """Обучает модель Wine и сохраняет модель вместе с метаданными."""

    wine = load_wine()
    x, y = wine.data, wine.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    cv_scores = cross_val_score(model, x, y, cv=5)

    metadata = {
        'feature_names': wine.feature_names,
        'target_names': ['Культивар 1', 'Культивар 2', 'Культивар 3'],
        'accuracy': float(accuracy),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'feature_ranges': {
            name: (float(x[:, index].min()), float(x[:, index].max()))
            for index, name in enumerate(wine.feature_names)
        },
    }

    model_dir = save_model_artifacts(model, metadata, 'wine')
    print(f'Wine Model Accuracy: {accuracy:.4f}')
    print(
        f'Wine CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})'
    )
    print(f'Model saved to {model_dir}')

    return model, metadata


if __name__ == '__main__':
    train_and_save_wine_model()
