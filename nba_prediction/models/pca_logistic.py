from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_prediction.config import RANDOM_SEED
from nba_prediction.models.spec import ModelSpec


def build_spec(feature_count):
    pca_components = [n for n in [3, 5, 7, 10] if n <= feature_count]
    return ModelSpec(
        name="PCA + Logistic Regression",
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("lr", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_SEED)),
            ]
        ),
        param_grid={"pca__n_components": pca_components, "lr__C": [0.01, 0.1, 1, 5]},
        threshold=0.40,
    )

