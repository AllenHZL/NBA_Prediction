from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_prediction.config import RANDOM_SEED
from nba_prediction.models.spec import ModelSpec


def build_spec(feature_count=None):
    return ModelSpec(
        name="Neural Network",
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(max_iter=3000, random_state=RANDOM_SEED)),
            ]
        ),
        param_grid={
            "mlp__hidden_layer_sizes": [(32,), (64, 32)],
            "mlp__alpha": [0.0001, 0.001, 0.01],
        },
        threshold=0.50,
    )

