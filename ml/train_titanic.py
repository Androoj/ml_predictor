from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from ml.common import save_model_artifacts


RANDOM_STATE = 42
TITANIC_DATA_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
FEATURE_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


def load_and_preprocess_data() -> tuple[pd.DataFrame, pd.Series]:
    """Загружает данные Titanic и преобразует выбранные признаки в числовой формат."""

    df = pd.read_csv(TITANIC_DATA_URL)
    df = df[['Survived', *FEATURE_COLUMNS]].copy()

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df = df.dropna()

    return df[FEATURE_COLUMNS], df['Survived']


def train_and_save_titanic_model() -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Обучает модель Titanic и сохраняет модель вместе с метаданными."""

    x, y = load_and_preprocess_data()

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
        min_samples_split=5,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    cv_scores = cross_val_score(model, x, y, cv=5)

    metadata = {
        'feature_names': [
            'Pclass (класс билета)',
            'Sex (0-женщина, 1-мужчина)',
            'Age (возраст)',
            'SibSp (братья/сестры/супруг)',
            'Parch (родители/дети)',
            'Fare (стоимость билета)',
            'Embarked (порт: 0-S, 1-C, 2-Q)',
        ],
        'target_names': ['Not Survived', 'Survived'],
        'accuracy': float(accuracy),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'feature_ranges': {
            column: (float(x[column].min()), float(x[column].max()))
            for column in FEATURE_COLUMNS
        },
    }

    model_dir = save_model_artifacts(model, metadata, 'titanic')
    print(f'Titanic Model Accuracy: {accuracy:.4f}')
    print(f'Titanic CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
    print(f'Model saved to {model_dir}')

    return model, metadata


if __name__ == '__main__':
    train_and_save_titanic_model()
