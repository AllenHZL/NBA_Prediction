from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
KAGGLE_DIR = PROJECT_DIR / "data" / "nba_kaggle_dataset"
IMAGE_DIR = PROJECT_DIR / "images"
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "output.md"

START_END_YEAR = 1997
LIVE_END_YEAR = 2026
COMPLETED_PLAYOFF_END_YEAR = 2025

TRAIN_START_YEAR = 2000
TRAIN_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
LIVE_SEASON = "2025-26"
BONUS_TEAMS = ("San Antonio Spurs", "New York Knicks")

RANDOM_SEED = 42
CV_SPLITS = 5

REGULAR_SEASON_GAME_TYPES = {"Regular Season", "NBA Emirates Cup"}
PLAYOFF_GAME_TYPES = {"Playoffs"}

MODEL_METRIC_COLUMNS = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "AUC",
    "TN",
    "FP",
    "FN",
    "TP",
]
