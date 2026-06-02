# Onboarding

這份 repo 的主流程很簡單：

```text
Kaggle raw CSV
  -> fetch_nba_data.py
  -> data/processed/*.csv
  -> run_nba_prediction.py
  -> logs/output.md + images/*.png + data/processed/*_comparison.csv
```

## How To Run

第一次準備資料，或 Kaggle raw data 有更新時：

```powershell
python fetch_nba_data.py --source kaggle
```

平常重新訓練、重新產生結果：

```powershell
python run_nba_prediction.py
```

## Read This Order

1. `README.md`
2. `docs/project_structure.md`
3. `docs/data_pipeline.md`
4. `docs/modeling.md`
5. `docs/research_findings.md`
6. `nba_prediction/pipeline.py`

## Where To Change Things

| 想改的內容 | 去哪裡改 |
|---|---|
| train/test 年份 | `nba_prediction/config.py` |
| bonus teams | `nba_prediction/config.py` |
| Kaggle raw data 轉換 | `nba_prediction/data/kaggle_source.py` |
| response / label 定義 | `nba_prediction/data/processed.py` |
| feature selection | `nba_prediction/data/processed.py` |
| 新增或修改模型 | `nba_prediction/models/` |
| 模型順序 | `nba_prediction/models/registry.py` |
| 評估指標 | `nba_prediction/evaluation/` |
| 圖片 | `nba_prediction/visualization/plots.py` |
| 完整流程 | `nba_prediction/pipeline.py` |

## Important Rules

- Features 只用 regular season stats。
- `2025-26` 只做 bonus prediction，不放進 train/test label。
- 圖片固定輸出到 `images/`。
- CLI 結果固定輸出到 `logs/output.md`。
- `data/processed/ranking_evaluation.csv` 是 final test ranking 評估。
- `data/processed/baseline_ranking_comparison.csv` 是模型與 random guess / W_PCT / NET_RATING 的增益比較。
- `data/processed/live_conference_finals_prediction.csv` 是 `2025-26` live 分區決賽隊伍預測。
- 改完程式後至少跑：

```powershell
python run_nba_prediction.py
```
