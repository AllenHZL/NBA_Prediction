# Research Findings

## 研究問題

本研究使用 NBA regular season team-level stats，預測球隊是否能進入分區決賽。

- Response：`is_conf_finalist`
- Positive class：該隊在該季季後賽進入分區決賽
- Features：只使用 regular season stats，避免 playoff leakage
- Train/CV：`2000-01` 到 `2020-21`
- Final test：`2021-22` 到 `2024-25`
- Live bonus：`2025-26`

## 主要模型結果

Final test 表現最好的模型是 Random Forest。

| Model | F1 | AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| Random Forest | 0.5098 | 0.8401 | 0.3714 | 0.8125 |
| Lasso Logistic Regression | 0.4815 | 0.8666 | 0.3421 | 0.8125 |
| PCA + Logistic Regression | 0.4706 | 0.8359 | 0.3429 | 0.7500 |
| Logistic Regression | 0.4483 | 0.8612 | 0.3095 | 0.8125 |
| Neural Network | 0.3333 | 0.7885 | 0.3571 | 0.3125 |

解讀：

- Random Forest 的 F1 最高，因此作為後續 ranking evaluation 與 bonus prediction 的主要模型。
- Recall 為 `0.8125`，代表模型能抓到多數實際進入分區決賽的球隊。
- Precision 為 `0.3714`，代表模型會多抓一些 false positives；這符合 NBA playoff prediction 的高不確定性。
- AUC 為 `0.8401`，表示模型對球隊競爭力排序有不錯的區辨能力。

## Ranking Evaluation

NBA 情境下，真正重要的問題不是單純判斷每支球隊 0/1，而是每季每分區選出最可能進入分區決賽的前 2 隊。

Final test ranking hit rate：

```text
Random Forest: 8/16 = 0.5000
```

也就是在 `2021-22` 到 `2024-25` 的 8 個分區中，總共 16 個分區決賽席次，模型命中 8 個。

## Baseline Comparison

老師建議不要只看準確率，而要呈現模型相較於亂猜的增益。因此我們加入三個 baseline：

| Method | Total Hits | Total Slots | Hit Rate | Perfect Conferences |
|---|---:|---:|---:|---:|
| Random Forest | 8 | 16 | 0.5000 | 1 |
| Random Guess Expected | 4 | 16 | 0.2500 | 0.2857 |
| W_PCT Top 2 | 6 | 16 | 0.3750 | 1 |
| NET_RATING Top 2 | 8 | 16 | 0.5000 | 2 |

重點解讀：

- Random Forest 的 hit rate 是 random guess 的 2 倍。
- Random Forest 高於只看 regular season 勝率的 baseline。
- Random Forest 與 `NET_RATING Top 2` 的 hit rate 相同，表示 net rating 本身就是非常強的 playoff competitiveness 訊號。
- 模型沒有明顯超越 net rating，但能整合勝率、效率值、投籃、籃板、失誤等多個 regular season features，提供更完整的機器學習比較框架。

## Neural Network 為何表現較差

Neural Network 在 training set 幾乎完全擬合，但 final test F1 只有 `0.3333`。

主要原因：

- 樣本數小，training samples 只有 `626` 筆。
- Positive class 稀少，每季 30 隊只有 4 隊會進入分區決賽。
- Team-level tabular data 通常較適合 Random Forest 或 Logistic Regression。
- Train F1 接近 `1.0000`，但 test F1 明顯下降，顯示 overfitting。

## 2025-26 Bonus Prediction

Bonus 1：2025-26 分區決賽隊伍預測。

| Conference | Rank | Team |
|---|---:|---|
| East | 1 | Detroit Pistons |
| East | 2 | New York Knicks |
| West | 1 | Oklahoma City Thunder |
| West | 2 | San Antonio Spurs |

這是模型原始研究目標的 live extension：使用 `2025-26` regular season features 預測東西區各自最可能進入分區決賽的前 2 隊。

Bonus 2：Spurs vs Knicks 總冠軍相對勝率。

```text
San Antonio Spurs: 0.5052
New York Knicks: 0.4948
```

這不是直接訓練總冠軍模型，而是把兩隊的 `p_conf_finalist_strength` 正規化後得到的衍生相對勝率。結果接近五五波，應解讀為模型認為兩隊競爭力非常接近。

## Limitations

- 沒有使用球員層級資料，例如傷病、明星球員缺陣、交易後陣容變化。
- 沒有納入 matchup effect，例如對位、防守風格、季後賽經驗。
- 沒有加入 playoff seed、home-court advantage、strength of schedule。
- 早期 NBA 與近年 NBA 的 pace、三分球使用率、play-in 制度不同，歷史資料可能存在時代差異。
- Bonus 冠軍預測是由分區決賽競爭力模型衍生，不是直接的 champion classifier。

## Slides 建議順序

1. 研究動機：季後賽預測與分區決賽競爭力。
2. Data：Kaggle team-level regular season / playoffs stats。
3. Response：`is_conf_finalist`，features 只用 regular season。
4. Models：Logistic、Lasso、Random Forest、PCA Logistic、Neural Network。
5. Model comparison：Random Forest F1 最高。
6. Ranking evaluation：每季每分區 playoff teams 取前 2。
7. Baseline comparison：模型相較 random guess 與 W_PCT 的增益。
8. Feature importance：`W_PCT`, `PIE`, `NET_RATING` 等。
9. Bonus prediction：2025-26 分區決賽與 Spurs vs Knicks。
10. Limitations 與 future work。
