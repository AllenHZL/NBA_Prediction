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
from nba_prediction.evaluation.ranking import evaluate_baseline_comparison, evaluate_ranking
from nba_prediction.logging_utils import tee_output_to_markdown
from nba_prediction.models.registry import get_model_specs
from nba_prediction.visualization.plots import (
    configure_matplotlib,
    plot_eda,
    plot_baseline_comparison,
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
    _run_feature_set_experiment(
        df_main=df_main,
        feature_set_label=None,
        output_prefix="",
        plot_eda_flag=True,
    )


def _run_feature_set_experiment(df_main, feature_set_label, output_prefix, plot_eda_flag):
    if feature_set_label:
        print("\n" + "#" * 100)
        print(feature_set_label)
        print("#" * 100)

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

    if plot_eda_flag:
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
    baseline_comparison_df = evaluate_baseline_comparison(df_test_res, ranking_model, ranking_df)

    roc_title = f"{feature_set_label}: ROC Curves" if feature_set_label else None
    importance_title = (
        f"{feature_set_label}: Random Forest Feature Importance (Top 15)"
        if feature_set_label
        else "Random Forest Feature Importance (Top 15)"
    )
    baseline_title = (
        f"{feature_set_label}: Model vs Baselines"
        if feature_set_label
        else "Conference Finals Ranking: Model vs Baselines"
    )
    plot_roc_curves(y_test, test_probs, filename=f"{output_prefix}roc_curves_comparison.png", title=roc_title)
    plot_feature_importance(
        trained_models.get("Random Forest"),
        feature_cols,
        filename=f"{output_prefix}feature_importance.png",
        title=importance_title,
    )
    plot_baseline_comparison(
        baseline_comparison_df,
        filename=f"{output_prefix}baseline_ranking_comparison.png",
        title=baseline_title,
    )

    final_model, final_medians = _fit_final_model(df_main, feature_cols, ranking_model, trained_models)
    live_prediction = _run_live_conference_finals_prediction(
        df_main,
        live_mask,
        feature_cols,
        final_model,
        final_medians,
        feature_set_label,
    )
    _run_bonus_prediction(live_prediction, feature_set_label)

    ranking_path = DATA_DIR / f"{output_prefix}ranking_evaluation.csv"
    baseline_path = DATA_DIR / f"{output_prefix}baseline_ranking_comparison.csv"
    live_path = DATA_DIR / f"{output_prefix}live_conference_finals_prediction.csv"
    model_path = DATA_DIR / f"{output_prefix}model_comparison.csv"
    ranking_df.to_csv(ranking_path, index=False)
    baseline_comparison_df.to_csv(baseline_path, index=False)
    live_prediction.to_csv(live_path, index=False)
    result_table.to_csv(model_path)
    print(f"Saved data/processed/{ranking_path.name}")
    print(f"Saved data/processed/{baseline_path.name}")
    print(f"Saved data/processed/{live_path.name}")
    print(f"Saved data/processed/{model_path.name}")


def _fit_final_model(df_main, feature_cols, ranking_model, trained_models):
    final_train_mask = (df_main["START_YEAR"] >= TRAIN_START_YEAR) & (df_main["START_YEAR"] <= TEST_END_YEAR)
    final_medians = df_main.loc[final_train_mask, feature_cols].median(numeric_only=True)
    X_final = fill_features(df_main.loc[final_train_mask], feature_cols, final_medians)
    y_final = df_main.loc[final_train_mask, "is_conf_finalist"].astype(int)
    final_model = clone(trained_models[ranking_model])
    final_model.fit(X_final, y_final)
    return final_model, final_medians


def _run_live_conference_finals_prediction(
    df_main,
    live_mask,
    feature_cols,
    final_model,
    final_medians,
    feature_set_label,
):
    live_df = df_main.loc[live_mask].copy()
    if live_df.empty:
        raise ValueError(f"Missing {LIVE_SEASON} regular season data for live prediction.")

    suffix = f" ({feature_set_label})" if feature_set_label else ""
    print("\n" + "=" * 100)
    print(f"{LIVE_SEASON} Conference Finals Prediction{suffix}")
    print("=" * 100)
    print(
        "This live prediction ranks all 30 teams from regular season features only; "
        f"{LIVE_SEASON} playoff data is not used."
    )

    X_live = fill_features(live_df, feature_cols, final_medians)
    live_df["p_conf_finalist_strength"] = final_model.predict_proba(X_live)[:, 1]
    live_df = live_df.sort_values(
        ["Conference", "p_conf_finalist_strength", "TEAM_NAME"],
        ascending=[True, False, True],
    )
    live_df["Rank"] = live_df.groupby("Conference").cumcount() + 1
    live_df["predicted_conf_finalist"] = live_df["Rank"] <= 2

    output_cols = [
        "SEASON",
        "Conference",
        "Rank",
        "TEAM_NAME",
        "W",
        "L",
        "W_PCT",
        "NET_RATING",
        "p_conf_finalist_strength",
        "predicted_conf_finalist",
    ]
    live_prediction = live_df[output_cols].copy()
    if len(live_prediction) != 30:
        raise ValueError(f"{LIVE_SEASON} live prediction should have 30 teams, got {len(live_prediction)}.")

    predicted_count = int(live_prediction["predicted_conf_finalist"].sum())
    if predicted_count != 4:
        raise ValueError(f"Expected 4 predicted conference finalists, got {predicted_count}.")

    for conf in ["East", "West"]:
        top2 = live_prediction[
            (live_prediction["Conference"] == conf) & (live_prediction["predicted_conf_finalist"])
        ].sort_values("Rank")
        if len(top2) != 2:
            raise ValueError(f"Expected 2 predicted {conf} finalists, got {len(top2)}.")
        print(f"\n{LIVE_SEASON} {conf} predicted conference finalists:")
        print(
            top2[
                [
                    "Rank",
                    "TEAM_NAME",
                    "W",
                    "L",
                    "W_PCT",
                    "NET_RATING",
                    "p_conf_finalist_strength",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    return live_prediction


def _run_bonus_prediction(live_prediction, feature_set_label):
    suffix = f" ({feature_set_label})" if feature_set_label else ""
    print("\n" + "=" * 100)
    print(f"Bonus prediction: 2025-26 NBA Finals, Spurs vs Knicks{suffix}")
    print("=" * 100)
    print(
        "This is a derived prediction from the conference-finalist strength model, "
        "not a model directly trained on NBA Finals champions."
    )
    missing_bonus = [team for team in BONUS_TEAMS if team not in set(live_prediction["TEAM_NAME"])]
    if missing_bonus:
        raise ValueError(f"Missing bonus teams in {LIVE_SEASON} data: {missing_bonus}")
    bonus = live_prediction[live_prediction["TEAM_NAME"].isin(BONUS_TEAMS)][
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
