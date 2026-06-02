from sklearn.ensemble import RandomForestClassifier

from nba_prediction.config import RANDOM_SEED
from nba_prediction.models.spec import ModelSpec


def build_spec(feature_count=None):
    return ModelSpec(
        name="Random Forest",
        estimator=RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [1, 3, 5],
        },
        threshold=0.40,
    )

