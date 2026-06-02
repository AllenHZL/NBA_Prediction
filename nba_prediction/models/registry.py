from nba_prediction.models import lasso, logistic, neural_network, pca_logistic, random_forest


def get_model_specs(feature_count):
    return [
        logistic.build_spec(feature_count),
        lasso.build_spec(feature_count),
        random_forest.build_spec(feature_count),
        pca_logistic.build_spec(feature_count),
        neural_network.build_spec(feature_count),
    ]

