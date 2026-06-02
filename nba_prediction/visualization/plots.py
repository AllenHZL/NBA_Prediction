import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from nba_prediction.config import IMAGE_DIR


def configure_matplotlib():
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_eda(df_main, analysis_mask):
    eda_features = [
        c
        for c in ["W_PCT", "PTS", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "EFG_PCT"]
        if c in df_main.columns
    ]
    corr_matrix = df_main.loc[analysis_mask, eda_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("NBA Team Stat Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "eda_correlation_heatmap.png", dpi=300)
    plt.close()
    print("Saved images/eda_correlation_heatmap.png")

    if "NET_RATING" in df_main.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            x="is_conf_finalist",
            y="NET_RATING",
            data=df_main.loc[analysis_mask],
            hue="is_conf_finalist",
            palette="Set2",
            legend=False,
        )
        plt.xticks([0, 1], ["Non-Finalist", "Conference Finalist"])
        plt.title("Net Rating Distribution by Conference Finalist Label", fontsize=14)
        plt.xlabel("Playoff Result")
        plt.ylabel("Net Rating")
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "eda_net_rating_dist.png", dpi=300)
        plt.close()
        print("Saved images/eda_net_rating_dist.png")


def plot_roc_curves(y_test, test_probs):
    plt.figure(figsize=(10, 8))
    for name, probs in test_probs.items():
        if len(set(y_test)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Regular Season Predicting Conference Finalists", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "roc_curves_comparison.png", dpi=300)
    plt.close()
    print("Saved images/roc_curves_comparison.png")


def plot_feature_importance(model, feature_cols):
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values(
        "Importance",
        ascending=False,
    )
    print("\nTop 15 Important Features (Random Forest)")
    print(importance_df.head(15).to_string(index=False))
    plt.figure(figsize=(12, 6))
    top15 = importance_df.head(15)
    sns.barplot(x="Importance", y="Feature", hue="Feature", data=top15, palette="viridis", legend=False)
    plt.title("Random Forest Feature Importance (Top 15)", fontsize=14)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=300)
    plt.close()
    print("Saved images/feature_importance.png")
    return importance_df

