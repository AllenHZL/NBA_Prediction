# Data Pipeline

## Source

目前主資料來源是本地 Kaggle dataset：

```text
data/nba_kaggle_dataset/
```

核心使用檔案：

- `Games.csv`：game id、日期、game type
- `TeamStatisticsExtended.csv`：team-game level traditional and advanced stats

`fetch_nba_data.py` 是資料更新入口，實際轉換邏輯在 `nba_prediction/data/kaggle_source.py`。

## Season Definition

NBA season 使用 `gameId` 推回 season start year，再轉成 `YYYY-YY` 格式。這比單純用日期可靠，因為 2019-20 bubble playoffs 延到 2020 年 9 月與 10 月，若只用月份會被錯分到 `2020-21`。

例子：

- `2019-20` playoffs 即使在 2020 年 9 月或 10 月舉行，仍歸到 `2019-20`
- `2025-26` regular season 保留作 bonus prediction
- `2025-26` playoffs 不寫入 completed playoff label

## Processed CSV

輸出位置：

```text
data/processed/
```

輸出檔：

- `team_stats_traditional_rs.csv`
- `team_stats_advanced_rs.csv`
- `team_stats_traditional_po.csv`
- `team_stats_advanced_po.csv`
- `metadata.json`

Regular season team stats 由 team-game rows 聚合成 team-season rows。Traditional stats 多數使用每場平均，勝敗紀錄優先使用資料中的 season record，必要時以逐場 win 欄位回推。

Playoff stats 同樣聚合成 team-season rows，但只包含已完成賽季到 `2024-25`。

## Validation Checks

資料更新時會檢查：

- `2025-26` regular season 有 30 隊
- `2024-25` playoffs 有 16 隊
- `2025-26` playoffs 不出現在 train/test label data
- 每個 completed season 剛好 4 支 `is_conf_finalist=1`
- 每個分區剛好 2 支 conference finalists
- 與舊 Brescou CSV 的 overlap season/team rows 能對齊

Root 目錄下的舊 CSV 僅用於 historical overlap validation，不是主訓練資料。
