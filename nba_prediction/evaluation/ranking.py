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

