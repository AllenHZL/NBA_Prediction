from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_prediction.config import RANDOM_SEED
from nba_prediction.models.spec import ModelSpec


def build_spec(feature_count=None):
    return ModelSpec(
        name="Lasso Logistic Regression",
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lasso",
                    LogisticRegression(
                        max_iter=5000,
                        penalty="l1",
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        param_grid={"lasso__C": [0.01, 0.1, 1, 5, 10]},
        threshold=0.40,
    )

