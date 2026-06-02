import argparse

from nba_prediction.data.kaggle_source import fetch_kaggle_dataset
from nba_prediction.data.validation import validate_outputs, write_outputs


def main():
    parser = argparse.ArgumentParser(description="Prepare NBA team-season data for the project.")
    parser.add_argument(
        "--source",
        choices=["auto", "kaggle"],
        default="auto",
        help="Data source to process. `auto` currently uses the local Kaggle dataset.",
    )
    args = parser.parse_args()

    source = "kaggle_local_eoinamoore"
    outputs = fetch_kaggle_dataset()
    validate_outputs(outputs)
    write_outputs(outputs, source)
    print(f"Done. Data source used: {source}")


if __name__ == "__main__":
    main()

