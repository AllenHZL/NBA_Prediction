import pandas as pd

from nba_prediction.config import (
    COMPLETED_PLAYOFF_END_YEAR,
    KAGGLE_DIR,
    LIVE_END_YEAR,
    PLAYOFF_GAME_TYPES,
    REGULAR_SEASON_GAME_TYPES,
    START_END_YEAR,
)
from nba_prediction.data.utils import (
    conference_for_team_id,
    make_team_name,
    mean_numeric,
    safe_divide,
    season_end_year_from_game_id,
    season_label,
    sum_numeric,
    to_numeric_columns,
    weighted_mean,
)


def choose_regular_season_record(group):
    row_count = len(group)
    wins_from_rows = int(pd.to_numeric(group["win"], errors="coerce").fillna(0).sum())
    losses_from_rows = row_count - wins_from_rows
    season_wins = pd.to_numeric(group.get("seasonWins"), errors="coerce").max()
    season_losses = pd.to_numeric(group.get("seasonLosses"), errors="coerce").max()
    if pd.notna(season_wins) and pd.notna(season_losses):
        season_games = int(season_wins + season_losses)
        if season_games >= row_count:
            return int(season_wins), int(season_losses), season_games
    return wins_from_rows, losses_from_rows, row_count


def choose_playoff_record(group):
    wins = int(pd.to_numeric(group["win"], errors="coerce").fillna(0).sum())
    games = len(group)
    return wins, games - wins, games


def load_kaggle_team_games(kaggle_dir=KAGGLE_DIR):
    required = ["Games.csv", "TeamStatisticsExtended.csv"]
    missing = [filename for filename in required if not (kaggle_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Kaggle dataset files in {kaggle_dir}: {missing}")

    games = pd.read_csv(
        kaggle_dir / "Games.csv",
        usecols=["gameId", "gameDate", "gameDateTimeEst", "gameType"],
    )
    games["GAME_DATE"] = pd.to_datetime(games["gameDate"], errors="coerce").fillna(
        pd.to_datetime(games["gameDateTimeEst"], errors="coerce")
    )
    games["SEASON_END_YEAR"] = [
        season_end_year_from_game_id(game_id, game_date)
        for game_id, game_date in zip(games["gameId"], games["GAME_DATE"])
    ]
    games["SEASON"] = games["SEASON_END_YEAR"].map(season_label)
    games = games.rename(columns={"gameType": "GAME_TYPE"})

    wanted_columns = {
        "gameId",
        "teamCity",
        "teamName",
        "teamId",
        "win",
        "teamScore",
        "opponentScore",
        "assists",
        "blocks",
        "steals",
        "fieldGoalsAttempted",
        "fieldGoalsMade",
        "fieldGoalsPercentage",
        "threePointersAttempted",
        "threePointersMade",
        "threePointersPercentage",
        "freeThrowsAttempted",
        "freeThrowsMade",
        "freeThrowsPercentage",
        "reboundsDefensive",
        "reboundsOffensive",
        "reboundsTotal",
        "foulsPersonal",
        "turnovers",
        "plusMinusPoints",
        "numMinutes",
        "seasonWins",
        "seasonLosses",
        "possessions",
        "estimatedOffensiveRating",
        "offensiveRating",
        "estimatedDefensiveRating",
        "defensiveRating",
        "estimatedNetRating",
        "netRating",
        "assistPercentage",
        "assistToTurnoverRatio",
        "assistRatio",
        "offensiveReboundPercentage",
        "defensiveReboundPercentage",
        "reboundPercentage",
        "teamTurnoverPercentage",
        "effectiveFieldGoalPercentage",
        "trueShootingPercentage",
        "estimatedPace",
        "pace",
        "pacePer40",
        "playerImpactEstimate",
        "opponentEffectiveFieldGoalPercentage",
        "opponentFreeThrowAttemptRate",
        "opponentTurnoverPercentage",
        "opponentOffensiveReboundPercentage",
    }
    team_games = pd.read_csv(
        kaggle_dir / "TeamStatisticsExtended.csv",
        usecols=lambda column: column in wanted_columns,
    )
    team_games = team_games.merge(
        games[["gameId", "GAME_TYPE", "GAME_DATE", "SEASON", "SEASON_END_YEAR"]],
        on="gameId",
        how="inner",
    )
    team_games = team_games[
        team_games["SEASON_END_YEAR"].between(START_END_YEAR, LIVE_END_YEAR)
        & team_games["GAME_TYPE"].isin(REGULAR_SEASON_GAME_TYPES | PLAYOFF_GAME_TYPES)
        & (pd.to_numeric(team_games["teamId"], errors="coerce") >= 1610610000)
    ].copy()
    team_games["TEAM_ID"] = team_games["teamId"].astype(int)
    team_games["TEAM_NAME"] = team_games.apply(make_team_name, axis=1)
    team_games["Conference"] = [
        conference_for_team_id(team_id, end_year)
        for team_id, end_year in zip(team_games["TEAM_ID"], team_games["SEASON_END_YEAR"])
    ]
    return team_games


def aggregate_traditional(games, season_type):
    rows = []
    group_keys = ["SEASON", "SEASON_END_YEAR", "TEAM_ID", "TEAM_NAME", "Conference"]
    for keys, group in games.groupby(group_keys, sort=True):
        season, _, team_id, team_name, conference = keys
        if season_type == "regular":
            wins, losses, gp = choose_regular_season_record(group)
        else:
            wins, losses, gp = choose_playoff_record(group)
        row_count = max(len(group), 1)
        fgm = sum_numeric(group, "fieldGoalsMade")
        fga = sum_numeric(group, "fieldGoalsAttempted")
        fg3m = sum_numeric(group, "threePointersMade")
        fg3a = sum_numeric(group, "threePointersAttempted")
        ftm = sum_numeric(group, "freeThrowsMade")
        fta = sum_numeric(group, "freeThrowsAttempted")
        rows.append(
            {
                "TEAM_ID": team_id,
                "TEAM_NAME": team_name,
                "SEASON": season,
                "Conference": conference,
                "GP": gp,
                "W": wins,
                "L": losses,
                "W_PCT": safe_divide(wins, wins + losses),
                "MIN": mean_numeric(group, "numMinutes"),
                "FGM": safe_divide(fgm, row_count),
                "FGA": safe_divide(fga, row_count),
                "FG_PCT": safe_divide(fgm, fga),
                "FG3M": safe_divide(fg3m, row_count),
                "FG3A": safe_divide(fg3a, row_count),
                "FG3_PCT": safe_divide(fg3m, fg3a),
                "FTM": safe_divide(ftm, row_count),
                "FTA": safe_divide(fta, row_count),
                "FT_PCT": safe_divide(ftm, fta),
                "OREB": safe_divide(sum_numeric(group, "reboundsOffensive"), row_count),
                "DREB": safe_divide(sum_numeric(group, "reboundsDefensive"), row_count),
                "REB": safe_divide(sum_numeric(group, "reboundsTotal"), row_count),
                "AST": safe_divide(sum_numeric(group, "assists"), row_count),
                "TOV": safe_divide(sum_numeric(group, "turnovers"), row_count),
                "STL": safe_divide(sum_numeric(group, "steals"), row_count),
                "BLK": safe_divide(sum_numeric(group, "blocks"), row_count),
                "PF": safe_divide(sum_numeric(group, "foulsPersonal"), row_count),
                "PTS": safe_divide(sum_numeric(group, "teamScore"), row_count),
                "PLUS_MINUS": mean_numeric(group, "plusMinusPoints"),
            }
        )
    return to_numeric_columns(pd.DataFrame(rows))


def aggregate_advanced(games, season_type):
    rows = []
    group_keys = ["SEASON", "SEASON_END_YEAR", "TEAM_ID", "TEAM_NAME", "Conference"]
    for keys, group in games.groupby(group_keys, sort=True):
        season, _, team_id, team_name, conference = keys
        if season_type == "regular":
            wins, losses, gp = choose_regular_season_record(group)
        else:
            wins, losses, gp = choose_playoff_record(group)
        fgm = sum_numeric(group, "fieldGoalsMade")
        fga = sum_numeric(group, "fieldGoalsAttempted")
        fg3m = sum_numeric(group, "threePointersMade")
        fta = sum_numeric(group, "freeThrowsAttempted")
        points = sum_numeric(group, "teamScore")
        opponent_points = sum_numeric(group, "opponentScore")
        possessions = sum_numeric(group, "possessions")
        off_rating = safe_divide(points * 100, possessions)
        def_rating = safe_divide(opponent_points * 100, possessions)
        rows.append(
            {
                "TEAM_ID": team_id,
                "TEAM_NAME": team_name,
                "SEASON": season,
                "Conference": conference,
                "GP": gp,
                "W": wins,
                "L": losses,
                "W_PCT": safe_divide(wins, wins + losses),
                "MIN": mean_numeric(group, "numMinutes"),
                "E_OFF_RATING": weighted_mean(group, "estimatedOffensiveRating"),
                "OFF_RATING": off_rating if pd.notna(off_rating) else weighted_mean(group, "offensiveRating"),
                "E_DEF_RATING": weighted_mean(group, "estimatedDefensiveRating"),
                "DEF_RATING": def_rating if pd.notna(def_rating) else weighted_mean(group, "defensiveRating"),
                "E_NET_RATING": weighted_mean(group, "estimatedNetRating"),
                "NET_RATING": (
                    off_rating - def_rating
                    if pd.notna(off_rating) and pd.notna(def_rating)
                    else weighted_mean(group, "netRating")
                ),
                "AST_PCT": weighted_mean(group, "assistPercentage"),
                "AST_TO": safe_divide(sum_numeric(group, "assists"), sum_numeric(group, "turnovers")),
                "AST_RATIO": weighted_mean(group, "assistRatio"),
                "OREB_PCT": weighted_mean(group, "offensiveReboundPercentage"),
                "DREB_PCT": weighted_mean(group, "defensiveReboundPercentage"),
                "REB_PCT": weighted_mean(group, "reboundPercentage"),
                "TM_TOV_PCT": weighted_mean(group, "teamTurnoverPercentage"),
                "EFG_PCT": safe_divide(fgm + 0.5 * fg3m, fga),
                "TS_PCT": safe_divide(points, 2 * (fga + 0.44 * fta)),
                "E_PACE": weighted_mean(group, "estimatedPace"),
                "PACE": weighted_mean(group, "pace"),
                "PACE_PER40": weighted_mean(group, "pacePer40"),
                "POSS": mean_numeric(group, "possessions"),
                "PIE": weighted_mean(group, "playerImpactEstimate"),
                "OPP_EFG_PCT": weighted_mean(group, "opponentEffectiveFieldGoalPercentage"),
                "OPP_FTA_RATE": weighted_mean(group, "opponentFreeThrowAttemptRate"),
                "OPP_TOV_PCT": weighted_mean(group, "opponentTurnoverPercentage"),
                "OPP_OREB_PCT": weighted_mean(group, "opponentOffensiveReboundPercentage"),
            }
        )
    return to_numeric_columns(pd.DataFrame(rows))


def fetch_kaggle_dataset(kaggle_dir=KAGGLE_DIR):
    print(f"Preparing local Kaggle dataset from {kaggle_dir}")
    team_games = load_kaggle_team_games(kaggle_dir)
    regular_games = team_games[team_games["GAME_TYPE"].isin(REGULAR_SEASON_GAME_TYPES)].copy()
    playoff_games = team_games[
        team_games["GAME_TYPE"].isin(PLAYOFF_GAME_TYPES)
        & (team_games["SEASON_END_YEAR"] <= COMPLETED_PLAYOFF_END_YEAR)
    ].copy()

    return {
        "team_stats_traditional_rs.csv": aggregate_traditional(regular_games, "regular"),
        "team_stats_advanced_rs.csv": aggregate_advanced(regular_games, "regular"),
        "team_stats_traditional_po.csv": aggregate_traditional(playoff_games, "playoffs"),
        "team_stats_advanced_po.csv": aggregate_advanced(playoff_games, "playoffs"),
    }

