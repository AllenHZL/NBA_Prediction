import pandas as pd
import numpy as np

from nba_prediction.config import DATA_DIR, LIVE_SEASON, TEST_END_YEAR
from nba_prediction.data.utils import season_start_year


def require_data_file(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python fetch_nba_data.py --source kaggle` first.")
    return path


def load_processed_data():
    print("Loading processed NBA data...")
    df_rs_trad = pd.read_csv(require_data_file("team_stats_traditional_rs.csv"))
    df_rs_adv = pd.read_csv(require_data_file("team_stats_advanced_rs.csv"))
    df_po_trad = pd.read_csv(require_data_file("team_stats_traditional_po.csv"))

    metadata_path = DATA_DIR / "metadata.json"
    if metadata_path.exists():
        print(f"Data metadata: {metadata_path}")
        print(metadata_path.read_text(encoding="utf-8"))

    return df_rs_trad, df_rs_adv, df_po_trad


def build_model_dataset(df_rs_trad, df_rs_adv, df_po_trad):
    merge_keys = ["TEAM_ID", "TEAM_NAME", "SEASON"]
    if "Conference" in df_rs_trad.columns and "Conference" in df_rs_adv.columns:
        merge_keys.append("Conference")

    dup_cols = [col for col in df_rs_adv.columns if col in df_rs_trad.columns and col not in merge_keys]
    df_rs = pd.merge(df_rs_trad, df_rs_adv.drop(columns=dup_cols), on=merge_keys, how="inner")

    df_po_label = df_po_trad[["TEAM_ID", "TEAM_NAME", "SEASON", "W"]].rename(columns={"W": "PO_WINS"})
    df_main = pd.merge(df_rs, df_po_label, on=["TEAM_ID", "TEAM_NAME", "SEASON"], how="left")
    df_main["IS_PLAYOFF_TEAM"] = df_main["PO_WINS"].notna()
    df_main["PO_WINS"] = df_main["PO_WINS"].fillna(0)
    df_main["START_YEAR"] = df_main["SEASON"].map(season_start_year)
    df_main["is_conf_finalist"] = df_main.apply(
        lambda row: int(row["PO_WINS"] >= (8 if row["START_YEAR"] >= 2002 else 7)),
        axis=1,
    )
    return df_main


def validate_labels(df_main):
    completed = df_main[(df_main["START_YEAR"] >= 2000) & (df_main["START_YEAR"] <= TEST_END_YEAR)]
    positives_by_season = completed.groupby("SEASON")["is_conf_finalist"].sum()
    bad = positives_by_season[positives_by_season != 4]
    if not bad.empty:
        raise ValueError(f"Label check failed; expected 4 conference finalists per season:\n{bad}")

    by_conf = completed.groupby(["SEASON", "Conference"])["is_conf_finalist"].sum()
    bad_conf = by_conf[by_conf != 2]
    if not bad_conf.empty:
        raise ValueError(f"Conference label check failed; expected 2 per conference:\n{bad_conf}")

    live_in_label = completed[completed["SEASON"] == LIVE_SEASON]
    if not live_in_label.empty:
        raise ValueError(f"{LIVE_SEASON} leaked into train/test label data.")

    print("Label checks passed: each completed season has 4 finalists, 2 per conference.")


def select_feature_columns(df_main, train_mask):
    exclude_cols = {
        "TEAM_ID",
        "TEAM_NAME",
        "SEASON",
        "PO_WINS",
        "is_conf_finalist",
        "GP",
        "Conference",
        "IS_PLAYOFF_TEAM",
        "START_YEAR",
    }
    numeric_cols = df_main.select_dtypes(include=[np.number]).columns
    candidates = [col for col in numeric_cols if col not in exclude_cols and not col.endswith("_RANK")]
    train_df = df_main.loc[train_mask, candidates]
    return [
        col
        for col in candidates
        if train_df[col].notna().any() and train_df[col].nunique(dropna=True) > 1
    ]


def fill_features(df, feature_cols, medians):
    return df[feature_cols].fillna(medians).fillna(0)

