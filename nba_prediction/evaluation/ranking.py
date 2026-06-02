from math import comb

import pandas as pd


def evaluate_ranking(df_result, model_name):
    print("\nRanking-based evaluation: playoff teams, top 2 per conference")
    rows = []
    prob_col = f"{model_name}_prob"
    for season in sorted(df_result["SEASON"].unique()):
        for conf in ["East", "West"]:
            pool = df_result[
                (df_result["SEASON"] == season)
                & (df_result["Conference"] == conf)
                & (df_result["IS_PLAYOFF_TEAM"])
            ].copy()
            if pool.empty:
                continue
            predicted = pool.sort_values(prob_col, ascending=False).head(2)["TEAM_NAME"].tolist()
            actual = pool[pool["is_conf_finalist"] == 1]["TEAM_NAME"].tolist()
            hits = len(set(predicted) & set(actual))
            rows.append(
                {
                    "SEASON": season,
                    "Conference": conf,
                    "Predicted Top 2": predicted,
                    "Actual Finalists": actual,
                    "Hits": hits,
                }
            )
            print(f"  {season} {conf}: predicted={predicted} | actual={actual} | hits={hits}/2")

    ranking_df = pd.DataFrame(rows)
    total_hits = int(ranking_df["Hits"].sum())
    total_slots = int(len(ranking_df) * 2)
    perfect_conf = int((ranking_df["Hits"] == 2).sum())
    print(f"Ranking hit rate: {total_hits}/{total_slots} = {total_hits / total_slots:.4f}")
    print(f"Perfect conference predictions: {perfect_conf}/{len(ranking_df)}")
    return ranking_df


def evaluate_baseline_comparison(df_result, model_name, ranking_df):
    print("\nBaseline comparison: final-test playoff teams, top 2 per conference")
    rows = []
    rows.append(_summarize_model_ranking(model_name, ranking_df))
    rows.append(_summarize_random_guess(df_result))
    rows.append(_summarize_score_baseline(df_result, "W_PCT Top 2", "W_PCT"))
    rows.append(_summarize_score_baseline(df_result, "NET_RATING Top 2", "NET_RATING"))

    comparison_df = pd.DataFrame(rows)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return comparison_df


def _ranking_pools(df_result):
    for season in sorted(df_result["SEASON"].unique()):
        for conf in ["East", "West"]:
            pool = df_result[
                (df_result["SEASON"] == season)
                & (df_result["Conference"] == conf)
                & (df_result["IS_PLAYOFF_TEAM"])
            ].copy()
            if not pool.empty:
                yield season, conf, pool


def _summarize_model_ranking(model_name, ranking_df):
    total_hits = float(ranking_df["Hits"].sum())
    total_slots = float(len(ranking_df) * 2)
    perfect_conf = float((ranking_df["Hits"] == 2).sum())
    return {
        "Method": model_name,
        "Total Hits": total_hits,
        "Total Slots": total_slots,
        "Hit Rate": total_hits / total_slots,
        "Perfect Conferences": perfect_conf,
    }


def _summarize_random_guess(df_result):
    expected_hits = 0.0
    expected_perfect = 0.0
    total_slots = 0.0
    for _, _, pool in _ranking_pools(df_result):
        n_teams = len(pool)
        n_actual = int(pool["is_conf_finalist"].sum())
        picks = 2
        expected_hits += picks * n_actual / n_teams
        expected_perfect += 1 / comb(n_teams, picks)
        total_slots += picks

    return {
        "Method": "Random Guess (Expected)",
        "Total Hits": expected_hits,
        "Total Slots": total_slots,
        "Hit Rate": expected_hits / total_slots,
        "Perfect Conferences": expected_perfect,
    }


def _summarize_score_baseline(df_result, method_name, score_col):
    if score_col not in df_result.columns:
        raise ValueError(f"Missing baseline score column: {score_col}")

    total_hits = 0.0
    total_slots = 0.0
    perfect_conf = 0.0
    for _, _, pool in _ranking_pools(df_result):
        predicted = pool.sort_values(score_col, ascending=False).head(2)["TEAM_NAME"].tolist()
        actual = pool[pool["is_conf_finalist"] == 1]["TEAM_NAME"].tolist()
        hits = len(set(predicted) & set(actual))
        total_hits += hits
        total_slots += 2
        perfect_conf += int(hits == 2)

    return {
        "Method": method_name,
        "Total Hits": total_hits,
        "Total Slots": total_slots,
        "Hit Rate": total_hits / total_slots,
        "Perfect Conferences": perfect_conf,
    }
