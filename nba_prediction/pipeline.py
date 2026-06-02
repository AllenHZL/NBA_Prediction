import warnings

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from nba_prediction.config import (
    BONUS_TEAMS,
    CV_SPLITS,
    DATA_DIR,
    IMAGE_DIR,
    LIVE_SEASON,
    LOG_FILE,
    MODEL_METRIC_COLUMNS,
    TEST_END_YEAR,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
    TRAIN_START_YEAR,
)
from nba_prediction.data.processed import (
    build_model_dataset,
    fill_features,
    load_processed_data,
    select_feature_columns,
    validate_labels,
)
from nba_prediction.evaluation.metrics import calculate_metrics, print_metrics
from nba_prediction.evaluation.ranking import evaluate_ranking
from nba_prediction.logging_utils import tee_output_to_markdown
from nba_prediction.models.registry import get_model_specs
from nba_prediction.visualization.plots import (
    configure_matplotlib,
    plot_eda,
    plot_feature_importance,
    plot_roc_curves,
)


def run_pipeline():
    with tee_output_to_markdown(LOG_FILE):
        warnings.filterwarnings("ignore")
        configure_matplotlib()
        IMAGE_DIR.mkdir(exist_ok=True)
        _run_pipeline()


def _run_pipeline():
    df_rs_trad, df_rs_adv, df_po_trad = load_processed_data()
    df_main = build_model_dataset(df_rs_trad, df_rs_adv, df_po_trad)
    validate_labels(df_main)

    analysis_mask = (df_main["START_YEAR"] >= TRAIN_START_YEAR) & (df_main["START_YEAR"] <= TEST_END_YEAR)
    train_mask = analysis_mask & (df_main["START_YEAR"] <= TRAIN_END_YEAR)
    test_mask = analysis_mask & (df_main["START_YEAR"] >= TEST_START_YEAR)
    live_mask = df_main["SEASON"] == LIVE_SEASON

    feature_cols = select_feature_columns(df_main, train_mask)
    medians = df_main.loc[train_mask, feature_cols].median(numeric_only=True)

    X_train = fill_features(df_main.loc[train_mask], feature_cols, medians)
    X_test = fill_features(df_main.loc[test_mask], feature_cols, medians)
    y_train = df_main.loc[train_mask, "is_conf_finalist"].astype(int)
    y_test = df_main.loc[test_mask, "is_conf_finalist"].astype(int)

    print(f"Training seasons: 2000-01 to {TRAIN_END_YEAR}-{str(TRAIN_END_YEAR + 1)[-2:]}")
    print(
        f"Testing seasons : {TEST_START_YEAR}-{str(TEST_START_YEAR + 1)[-2:]} "
        f"to {TEST_END_YEAR}-{str(TEST_END_YEAR + 1)[-2:]}"
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")
    print(f"Feature count   : {len(feature_cols)}")
    print(f"Features        : {feature_cols}")

    plot_eda(df_main, analysis_mask)

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    df_test_res = df_main.loc[test_mask].copy()
    model_results = {}
    test_probs = {}
    trained_models = {}

    for spec in get_model_specs(len(feature_cols)):
        print("\n" + "=" * 100)
        print(f"Training model: {spec.name}")
        print("=" * 100)
        grid = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            cv=tscv,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        trained_models[spec.name] = best_model
        print(f"Best params: {grid.best_params_}")
        print(f"Best CV F1 : {grid.best_score_:.4f}")

        train_prob = best_model.predict_proba(X_train)[:, 1]
        test_prob = best_model.predict_proba(X_test)[:, 1]
        train_pred = (train_prob > spec.threshold).astype(int)
        test_pred = (test_prob > spec.threshold).astype(int)

        train_metrics = calculate_metrics(y_train, train_pred, train_prob)
        test_metrics = calculate_metrics(y_test, test_pred, test_prob)
        print_metrics("Train Set", train_metrics)
        print("-" * 80)
        print_metrics("Final Test Set", test_metrics)

        model_results[spec.name] = test_metrics
        test_probs[spec.name] = test_prob
        df_test_res[f"{spec.name}_prob"] = test_prob
        df_test_res[f"{spec.name}_pred"] = test_pred

    result_table = pd.DataFrame(model_results).T[MODEL_METRIC_COLUMNS].sort_values(
        ["F1", "AUC"],
        ascending=False,
    )
    print("\nModel comparison on final test set:")
    print(result_table.to_string(float_format=lambda x: f"{x:.4f}"))

    ranking_model = result_table.index[0]
    print(f"\nRanking evaluation uses best final-test F1 model: {ranking_model}")
    ranking_df = evaluate_ranking(df_test_res, ranking_model)

    plot_roc_curves(y_test, test_probs)
    plot_feature_importance(trained_models.get("Random Forest"), feature_cols)

    _run_bonus_prediction(df_main, live_mask, feature_cols, ranking_model, trained_models)

    ranking_df.to_csv(DATA_DIR / "ranking_evaluation.csv", index=False)
    result_table.to_csv(DATA_DIR / "model_comparison.csv")
    print("Saved data/processed/ranking_evaluation.csv")
    print("Saved data/processed/model_comparison.csv")


def _run_bonus_prediction(df_main, live_mask, feature_cols, ranking_model, trained_models):
    print("\n" + "=" * 100)
    print("Bonus prediction: 2025-26 NBA Finals, Spurs vs Knicks")
    print("=" * 100)
    print(
        "This is a derived prediction from the conference-finalist strength model, "
        "not a model directly trained on NBA Finals champions."
    )
    final_train_mask = (df_main["START_YEAR"] >= TRAIN_START_YEAR) & (df_main["START_YEAR"] <= TEST_END_YEAR)
    final_medians = df_main.loc[final_train_mask, feature_cols].median(numeric_only=True)
    X_final = fill_features(df_main.loc[final_train_mask], feature_cols, final_medians)
    y_final = df_main.loc[final_train_mask, "is_conf_finalist"].astype(int)
    final_model = clone(trained_models[ranking_model])
    final_model.fit(X_final, y_final)

    live_df = df_main.loc[live_mask].copy()
    missing_bonus = [team for team in BONUS_TEAMS if team not in set(live_df["TEAM_NAME"])]
    if missing_bonus:
        raise ValueError(f"Missing bonus teams in {LIVE_SEASON} data: {missing_bonus}")
    X_live = fill_features(live_df, feature_cols, final_medians)
    live_df["p_conf_finalist_strength"] = final_model.predict_proba(X_live)[:, 1]
    bonus = live_df[live_df["TEAM_NAME"].isin(BONUS_TEAMS)][
        ["TEAM_NAME", "W", "L", "W_PCT", "NET_RATING", "p_conf_finalist_strength"]
    ].sort_values("TEAM_NAME")
    print(bonus.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    probs = bonus.set_index("TEAM_NAME")["p_conf_finalist_strength"]
    p_spurs = probs["San Antonio Spurs"]
    p_knicks = probs["New York Knicks"]
    total = p_spurs + p_knicks
    spurs_title_prob = p_spurs / total if total else 0.5
    knicks_title_prob = p_knicks / total if total else 0.5
    print(f"Derived Finals probability - San Antonio Spurs: {spurs_title_prob:.4f}")
    print(f"Derived Finals probability - New York Knicks    : {knicks_title_prob:.4f}")

