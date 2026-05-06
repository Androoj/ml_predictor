from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import Any

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from ml.common import save_model_artifacts


RANDOM_STATE = 42


def train_and_save_iris_model() -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Обучает модель Iris и сохраняет модель вместе с метаданными."""

    iris = load_iris()
    x, y = iris.data, iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    cv_scores = cross_val_score(model, x, y, cv=5)

    metadata = {
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'accuracy': float(accuracy),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'feature_ranges': {
            name: (float(x[:, index].min()), float(x[:, index].max()))
            for index, name in enumerate(iris.feature_names)
        },
    }

    model_dir = save_model_artifacts(model, metadata, 'iris')
    print(f'Iris Model Accuracy: {accuracy:.4f}')
    print(f'Iris CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
    print(f'Model saved to {model_dir}')

    return model, metadata


if __name__ == '__main__':
    train_and_save_iris_model()
