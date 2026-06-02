# Modeling

## Prediction Target

主要 response 是：

```text
is_conf_finalist
```

定義：

- `2002-03` 之後：playoff wins `>= 8`
- `2001-02` 以前：playoff wins `>= 7`

這個門檻能對應到進入分區決賽的球隊數：每季 4 隊，每個分區 2 隊。

## Feature Policy

Features 只使用 regular season team stats，避免用到 playoff 資訊造成 leakage。

排除欄位：

- identifiers：`TEAM_ID`, `TEAM_NAME`, `SEASON`
- labels：`PO_WINS`, `is_conf_finalist`
- grouping/status：`Conference`, `IS_PLAYOFF_TEAM`, `START_YEAR`
- volume-only control：`GP`
- NBA.com rank columns：所有 `_RANK` 欄位

缺失值使用 training set median 補值，再補 0 作保底。

## Methods

模型設定集中在 `nba_prediction/models/`，每個方法回傳同一種 `ModelSpec`：

- `Logistic Regression`
- `Lasso Logistic Regression`
- `Random Forest`
- `PCA + Logistic Regression`
- `Neural Network`

`models/registry.py` 固定模型順序，讓每次執行的比較表與 log 順序穩定。

## Evaluation

資料切分：

- train/CV：`2000-01` 到 `2020-21`
- final test：`2021-22` 到 `2024-25`
- live bonus：`2025-26`

Binary metrics：

- Accuracy
- Precision
- Recall
- Specificity
- F1
- ROC-AUC
- Confusion matrix

NBA 情境 ranking metric：

- 對 final test 的每個 season/conference
- 只在 playoff teams 中排序
- 取模型機率前 2 名
- 計算與實際分區決賽隊伍的 hits

## Bonus Prediction

Bonus 使用完成賽季 `2000-01` 到 `2024-25` 重新訓練 final model，再對 `2025-26` Spurs 和 Knicks regular season features 預測：

```text
p_conf_finalist_strength
```

相對冠軍機率：

```text
P(Spurs) = p_spurs / (p_spurs + p_knicks)
P(Knicks) = p_knicks / (p_spurs + p_knicks)
```

這是由分區決賽競爭力模型衍生的 bonus prediction，不是直接以 NBA champion 作為 response 的總冠軍模型。

