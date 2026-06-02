import json

import numpy as np
import pandas as pd

from nba_prediction.config import COMPLETED_PLAYOFF_END_YEAR, DATA_DIR, PROJECT_DIR, START_END_YEAR, LIVE_END_YEAR
from nba_prediction.data.utils import normalize_team_name_for_overlap, season_label


def validate_outputs(outputs):
    rs = outputs["team_stats_traditional_rs.csv"]
    po = outputs["team_stats_traditional_po.csv"]
    rs_2026 = rs[rs["SEASON"] == "2025-26"]
    po_2025 = po[po["SEASON"] == "2024-25"]
    if len(rs_2026) != 30:
        raise ValueError(f"2025-26 regular season should have 30 teams, got {len(rs_2026)}")
    if "2025-26" in set(po["SEASON"]):
        raise ValueError("2025-26 playoffs must not be included in train/test label data.")
    if len(po_2025) != 16:
        raise ValueError(f"2024-25 playoffs should have 16 teams, got {len(po_2025)}")

    labels = po.copy()
    labels["START_YEAR"] = labels["SEASON"].str[:4].astype(int)
    labels["is_conf_finalist"] = np.where(labels["START_YEAR"] >= 2002, labels["W"] >= 8, labels["W"] >= 7)
    completed = labels[(labels["START_YEAR"] >= 2000) & (labels["START_YEAR"] <= COMPLETED_PLAYOFF_END_YEAR - 1)]
    bad_seasons = completed.groupby("SEASON")["is_conf_finalist"].sum()
    bad_seasons = bad_seasons[bad_seasons != 4]
    if not bad_seasons.empty:
        raise ValueError(f"Expected exactly 4 conference finalists per season:\n{bad_seasons}")
    bad_conferences = completed.groupby(["SEASON", "Conference"])["is_conf_finalist"].sum()
    bad_conferences = bad_conferences[bad_conferences != 2]
    if not bad_conferences.empty:
        raise ValueError(f"Expected exactly 2 finalists per conference:\n{bad_conferences}")
    print(
        "Fetch checks passed: 2025-26 RS has 30 teams; 2024-25 PO has 16 teams; "
        "labels have 4 finalists and 2 per conference."
    )


def historical_overlap_report(outputs):
    old_rs_path = PROJECT_DIR / "team_stats_traditional_rs.csv"
    old_adv_path = PROJECT_DIR / "team_stats_advanced_rs.csv"
    if not old_rs_path.exists() or not old_adv_path.exists():
        return {"status": "skipped", "reason": "old CSV files not found"}

    def norm(df):
        df = df.copy()
        df["TEAM_NAME_NORM"] = df["TEAM_NAME"].map(normalize_team_name_for_overlap)
        return df

    old_rs = norm(pd.read_csv(old_rs_path))
    old_adv = norm(pd.read_csv(old_adv_path))
    new_rs = norm(outputs["team_stats_traditional_rs.csv"])
    new_adv = norm(outputs["team_stats_advanced_rs.csv"])

    seasons = sorted(set(old_rs["SEASON"]) & set(new_rs["SEASON"]))
    old_rs = old_rs[old_rs["SEASON"].isin(seasons)]
    new_rs = new_rs[new_rs["SEASON"].isin(seasons)]
    merged_rs = old_rs.merge(new_rs, on=["SEASON", "TEAM_NAME_NORM"], suffixes=("_old", "_new"))

    old_adv = old_adv[old_adv["SEASON"].isin(seasons)]
    new_adv = new_adv[new_adv["SEASON"].isin(seasons)]
    merged_adv = old_adv.merge(new_adv, on=["SEASON", "TEAM_NAME_NORM"], suffixes=("_old", "_new"))

    report = {
        "status": "completed",
        "overlap_seasons": [seasons[0], seasons[-1]],
        "old_rs_rows": int(len(old_rs)),
        "new_rs_rows": int(len(new_rs)),
        "matched_rs_rows": int(len(merged_rs)),
        "max_abs_w_diff": float((merged_rs["W_old"] - merged_rs["W_new"]).abs().max()),
        "max_abs_l_diff": float((merged_rs["L_old"] - merged_rs["L_new"]).abs().max()),
        "max_abs_w_pct_diff": float((merged_rs["W_PCT_old"] - merged_rs["W_PCT_new"]).abs().max()),
    }
    if "NET_RATING_old" in merged_adv.columns and "NET_RATING_new" in merged_adv.columns:
        report["max_abs_net_rating_diff"] = float(
            (merged_adv["NET_RATING_old"] - merged_adv["NET_RATING_new"]).abs().max()
        )
    return report


def write_outputs(outputs, source):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, df in outputs.items():
        df = df.sort_values(["SEASON", "TEAM_NAME"]).reset_index(drop=True)
        df.to_csv(DATA_DIR / filename, index=False)
        print(f"Wrote {filename}: {df.shape}")
    metadata = {
        "source": source,
        "regular_seasons": [season_label(START_END_YEAR), season_label(LIVE_END_YEAR)],
        "completed_playoff_seasons": [
            season_label(START_END_YEAR),
            season_label(COMPLETED_PLAYOFF_END_YEAR),
        ],
        "note": "2025-26 playoffs are intentionally excluded from training/test labels.",
        "historical_overlap": historical_overlap_report(outputs),
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote metadata.json")

