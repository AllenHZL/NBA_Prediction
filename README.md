# NBA Prediction

以 NBA regular season team-level stats 預測球隊是否能進入分區決賽，並用同一個「分區決賽競爭力」模型衍生 2025-26 Spurs vs Knicks 的總冠軍 bonus prediction。

## Project Goal

主要 supervised classification 目標：

- Response：`is_conf_finalist`
- Positive class：該隊在該季季後賽進入分區決賽
- Feature：只使用 regular season team stats，避免 playoff leakage

Bonus prediction：

- 使用完成賽季 `2000-01` 到 `2024-25` 重新訓練最佳模型
- 對 `2025-26` Spurs 和 Knicks regular season features 預測 `p_conf_finalist_strength`
- 用兩隊 strength 正規化成相對總冠軍機率
- 這不是直接訓練總冠軍模型

## Setup

```powershell
conda create -n nba_prediction python=3.11 -y
conda activate nba_prediction
pip install -r requirements.txt
```

## Data

主要資料來源是本地 Kaggle dataset，放在：

```text
data/nba_kaggle_dataset/
```

必要檔案：

- `Games.csv`
- `TeamStatisticsExtended.csv`

Root 目錄下的舊 Brescou CSV 只保留用來做 historical overlap check，不作為主要訓練資料。

## Run

先產生 processed CSV：

```powershell
python fetch_nba_data.py --source kaggle
```

再跑完整訓練、評估與圖表：

```powershell
python run_nba_prediction.py
```

## Outputs

- CLI log：`logs/output.md`
- EDA heatmap：`images/eda_correlation_heatmap.png`
- Net Rating distribution：`images/eda_net_rating_dist.png`
- ROC comparison：`images/roc_curves_comparison.png`
- Feature importance：`images/feature_importance.png`
- Model comparison：`data/processed/model_comparison.csv`
- Ranking evaluation：`data/processed/ranking_evaluation.csv`

## Documentation

- [Onboarding](docs/onboarding.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Modeling](docs/modeling.md)
- [Project Structure](docs/project_structure.md)
