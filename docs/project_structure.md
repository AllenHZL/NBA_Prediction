# Project Structure

## Entry Points

```text
run_nba_prediction.py
fetch_nba_data.py
```

`run_nba_prediction.py` 是完整訓練與評估入口，只負責呼叫 training pipeline。

`fetch_nba_data.py` 保留原本資料更新方式，但現在只負責 CLI parsing、呼叫 Kaggle data processing、validation 和寫出 processed CSV。

## Package Layout

```text
nba_prediction/
  config.py
  pipeline.py
  logging_utils.py
  data/
  models/
  evaluation/
  visualization/
```

`config.py`：

- 專案路徑
- season split
- live season
- bonus teams
- random seed
- output file names

`pipeline.py`：

- 串接 load data、build label、train models、evaluate、plot、bonus prediction
- 不放 Kaggle 原始資料解析細節
- 不放單一模型內部設定

`data/`：

- `kaggle_source.py`：Kaggle raw CSV 到 processed CSV
- `processed.py`：讀取 processed CSV、建 response、feature selection
- `validation.py`：資料完整性檢查與 metadata
- `utils.py`：season、team name、conference、數值聚合 helper

`models/`：

- 每個方法一個 `.py`
- 每個方法回傳 `ModelSpec`
- `registry.py` 固定模型清單與順序

`evaluation/`：

- binary metrics
- ranking-based evaluation

`visualization/`：

- EDA plots
- ROC curves
- feature importance

## Data And Outputs

```text
data/nba_kaggle_dataset/   raw Kaggle files
data/processed/            processed CSV, metadata, evaluation CSV
images/                    generated plots
logs/output.md             CLI output log
docs/                      project documentation
```

Root 目錄舊 Brescou CSV 仍保留，用來和 processed regular season data 做 overlap validation。
