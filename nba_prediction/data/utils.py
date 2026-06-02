import re

import numpy as np
import pandas as pd

from nba_prediction.data.constants import EAST_TEAM_IDS, WEST_TEAM_IDS


def season_label(end_year):
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def season_start_year(season):
    return int(str(season).split("-")[0])


def season_end_year_from_game_id(game_id, game_date):
    """NBA game ids encode the season start year; this avoids 2020 bubble date drift."""
    game_id_text = str(int(game_id)).zfill(8)
    season_start_yy = int(game_id_text[1:3])
    date = pd.to_datetime(game_date)
    season_start = (date.year // 100) * 100 + season_start_yy
    if season_start > date.year:
        season_start -= 100
    if date.month >= 9 and season_start < date.year - 1:
        season_start += 100
    return season_start + 1


def clean_team_name(value):
    name = str(value).replace("\xa0", " ").strip()
    name = name.replace("?", "")
    name = re.sub(r"\*.*$", "", name)
    name = re.sub(r"\s+\(\d+\)$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def normalize_team_name_for_overlap(value):
    return {
        "LA Clippers": "Los Angeles Clippers",
        "Oklahoma City Hornets": "New Orleans/Oklahoma City Hornets",
    }.get(clean_team_name(value), clean_team_name(value))


def conference_for_team_id(team_id, season_end_year):
    team_id = int(team_id)
    season_end_year = int(season_end_year)
    if team_id == 1610612740:
        return "East" if season_end_year <= 2004 else "West"
    if team_id in EAST_TEAM_IDS:
        return "East"
    if team_id in WEST_TEAM_IDS:
        return "West"
    raise ValueError(f"Missing conference mapping for TEAM_ID={team_id}")


def to_numeric_columns(df, exclude=("TEAM_NAME", "SEASON", "Conference")):
    df = df.copy()
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_divide(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def sum_numeric(group, column):
    if column not in group.columns:
        return np.nan
    return pd.to_numeric(group[column], errors="coerce").sum()


def mean_numeric(group, column):
    if column not in group.columns:
        return np.nan
    return pd.to_numeric(group[column], errors="coerce").mean()


def weighted_mean(group, value_column, weight_column="possessions"):
    if value_column not in group.columns:
        return np.nan
    values = pd.to_numeric(group[value_column], errors="coerce")
    if weight_column not in group.columns:
        return values.mean()
    weights = pd.to_numeric(group[weight_column], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return values.mean()
    return np.average(values[valid], weights=weights[valid])


def make_team_name(row):
    city = "" if pd.isna(row.get("teamCity")) else str(row.get("teamCity")).strip()
    name = "" if pd.isna(row.get("teamName")) else str(row.get("teamName")).strip()
    return clean_team_name(f"{city} {name}".strip())

