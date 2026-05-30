from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MERGED_PATH = ROOT / "merged_smart_failure.csv"
DATA_ROOT = ROOT.parent / "数据集" / "dcbrain" / "ssd_open_data"
SMART_RAW_PATH = DATA_ROOT / "smart_log_20191231.csv" / "20191231.csv"
FAILURE_TAG_PATH = DATA_ROOT / "ssd_failure_tag.csv" / "ssd_failure_tag.csv"
LOCATION_INFO_PATH = DATA_ROOT / "location_info_of_ssd.csv" / "location_info_of_ssd.csv"
DEDUP_PATH = ROOT / "dedup_smart_failure.csv"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"

FEATURES = ["r_5", "r_9", "r_12", "r_175", "n_5", "n_9", "n_12", "n_175"]
RAW_FEATURES = ["r_5", "r_9", "r_12", "r_175"]
CATEGORICAL_FEATURES = ["model"]
LOCATION_MODEL_FEATURES = ["app", "slot_id_grouped"]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)


def load_and_dedup() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    old_merged = pd.read_csv(MERGED_PATH)
    smart = pd.read_csv(SMART_RAW_PATH, usecols=["disk_id", "ds", "model", *FEATURES])
    failure_tag = pd.read_csv(FAILURE_TAG_PATH, usecols=["disk_id", "model", "failure"])
    location = pd.read_csv(
        LOCATION_INFO_PATH,
        usecols=["disk_id", "model", "app", "rack_id", "node_id", "slot_id"],
    )

    old_counts = old_merged["failure"].value_counts().sort_index().to_dict()
    smart_key_count = int(smart.drop_duplicates(["model", "disk_id"]).shape[0])
    failure_key_count = int(failure_tag.drop_duplicates(["model", "disk_id"]).shape[0])

    failure_tag = failure_tag.drop_duplicates(["model", "disk_id"])
    merged = smart.merge(failure_tag, on=["model", "disk_id"], how="left")
    merged["failure"] = merged["failure"].fillna(0).astype(int)

    dedup = merged.drop_duplicates(subset=["model", "disk_id"], keep="last").reset_index(drop=True)
    location = location.drop_duplicates(["model", "disk_id"])
    dedup = dedup.merge(location, on=["model", "disk_id"], how="left", indicator="location_merge")
    location_matched = int((dedup["location_merge"] == "both").sum())
    dedup = dedup.drop(columns=["location_merge"])
    dedup["app"] = dedup["app"].fillna("unknown")
    dedup["slot_id_cat"] = dedup["slot_id"].where(dedup["slot_id"].notna(), "missing").astype(str)
    slot_counts = dedup["slot_id_cat"].value_counts()
    common_slots = set(slot_counts[slot_counts >= 100].index)
    dedup["slot_id_grouped"] = dedup["slot_id_cat"].where(
        dedup["slot_id_cat"].isin(common_slots),
        "rare_slot",
    )
    dedup.to_csv(DEDUP_PATH, index=False)

    after_counts = dedup["failure"].value_counts().sort_index().to_dict()
    summary = {
        "a_merged_rows": int(len(old_merged)),
        "a_merged_unique_disk_id": int(old_merged["disk_id"].nunique()),
        "a_merged_failure_counts": {str(k): int(v) for k, v in old_counts.items()},
        "raw_smart_rows": int(len(smart)),
        "raw_smart_unique_disk_id": int(smart["disk_id"].nunique()),
        "raw_smart_unique_model_disk": smart_key_count,
        "failure_tag_rows": int(len(failure_tag)),
        "failure_tag_unique_model_disk": failure_key_count,
        "location_rows": int(len(location)),
        "location_unique_model_disk": int(location.drop_duplicates(["model", "disk_id"]).shape[0]),
        "location_matched_rows": location_matched,
        "location_match_rate": round(location_matched / max(len(dedup), 1), 6),
        "after_rows": int(len(dedup)),
        "after_unique_disk_id": int(dedup["disk_id"].nunique()),
        "after_unique_model_disk": int(dedup.drop_duplicates(["model", "disk_id"]).shape[0]),
        "after_failure_counts": {str(k): int(v) for k, v in after_counts.items()},
        "a_merge_extra_rows_due_to_disk_only_join": int(len(old_merged) - len(smart)),
        "a_merge_false_positive_failure_rows_vs_model_disk_join": int(
            old_counts.get(1, 0) - after_counts.get(1, 0)
        ),
        "model_disk_healthy_to_failed_ratio": round(
            after_counts.get(0, 0) / max(after_counts.get(1, 1), 1), 3
        ),
        "missing_pct_after_dedup": {
            col: round(float(val), 4)
            for col, val in (dedup[FEATURES].isna().mean() * 100).items()
        },
        "location_missing_pct": {
            col: round(float(val), 4)
            for col, val in (dedup[["app", "rack_id", "node_id", "slot_id"]].isna().mean() * 100).items()
        },
    }
    return merged, dedup, summary


def save_smart_boxplots(dedup: pd.DataFrame) -> None:
    plot_data = dedup[["failure"] + RAW_FEATURES].copy()
    for col in RAW_FEATURES:
        plot_data[col] = np.log1p(plot_data[col].clip(lower=0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, col in zip(axes, ["r_9", "r_12", "r_175"]):
        sns.boxplot(
            data=plot_data,
            x="failure",
            y=col,
            hue="failure",
            palette={0: "#4C78A8", 1: "#F58518"},
            showfliers=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{col} distribution by failure")
        ax.set_xlabel("failure (0=healthy, 1=failed)")
        ax.set_ylabel(f"log1p({col})")
    fig.suptitle("Other SMART Raw Indicators after Disk-level Deduplication", fontsize=14)
    fig.savefig(FIG_DIR / "smart_other_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_interpretable_smart_plots(dedup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meanings = {
        "r_5": "Reallocated sector count",
        "r_9": "Power on hours",
        "r_12": "Power cycle count",
        "r_175": "Power loss protection failure",
    }
    for feature in RAW_FEATURES:
        healthy = dedup.loc[dedup["failure"] == 0, feature]
        failed = dedup.loc[dedup["failure"] == 1, feature]
        healthy_mean = float(healthy.mean())
        failed_mean = float(failed.mean())
        rows.append(
            {
                "feature": feature,
                "meaning": meanings[feature],
                "healthy_mean": healthy_mean,
                "failed_mean": failed_mean,
                "failed_to_healthy_mean_ratio": failed_mean / healthy_mean if healthy_mean else np.nan,
                "healthy_nonzero_rate": float((healthy.fillna(0) > 0).mean()),
                "failed_nonzero_rate": float((failed.fillna(0) > 0).mean()),
                "missing_pct": float(dedup[feature].isna().mean() * 100),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "smart_feature_interpretation_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    sns.barplot(
        data=summary,
        x="feature",
        y="failed_to_healthy_mean_ratio",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].axhline(1, color="#D62728", linestyle="--", linewidth=1.2)
    axes[0].set_title("Failed / Healthy Mean Ratio")
    axes[0].set_xlabel("SMART feature")
    axes[0].set_ylabel("ratio (log scale)")
    axes[0].set_yscale("log")

    nonzero = summary.melt(
        id_vars=["feature"],
        value_vars=["healthy_nonzero_rate", "failed_nonzero_rate"],
        var_name="group",
        value_name="nonzero_rate",
    )
    nonzero["group"] = nonzero["group"].map(
        {"healthy_nonzero_rate": "healthy", "failed_nonzero_rate": "failed"}
    )
    sns.barplot(
        data=nonzero,
        x="feature",
        y="nonzero_rate",
        hue="group",
        palette={"healthy": "#4C78A8", "failed": "#F58518"},
        ax=axes[1],
    )
    axes[1].set_title("Non-zero Rate by Failure Status")
    axes[1].set_xlabel("SMART feature")
    axes[1].set_ylabel("non-zero rate")

    sns.barplot(
        data=summary,
        x="feature",
        y="missing_pct",
        hue="feature",
        palette="mako",
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title("Missing Rate")
    axes[2].set_xlabel("SMART feature")
    axes[2].set_ylabel("missing (%)")

    fig.suptitle("SMART Indicator Interpretability Summary", fontsize=14)
    fig.savefig(FIG_DIR / "smart_indicator_interpretation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    model_r5 = (
        dedup.assign(r5_nonzero=dedup["r_5"].fillna(0) > 0)
        .groupby("model")
        .agg(
            sample_count=("failure", "size"),
            failure_rate=("failure", "mean"),
            r5_nonzero_rate=("r5_nonzero", "mean"),
            r5_mean=("r_5", "mean"),
            r5_p95=("r_5", lambda x: x.quantile(0.95)),
        )
        .reset_index()
        .sort_values("failure_rate", ascending=False)
    )
    model_r5.to_csv(RESULT_DIR / "r5_model_signal_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    sns.barplot(
        data=model_r5,
        x="model",
        y="failure_rate",
        hue="model",
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Failure Rate by Model")
    axes[0].set_xlabel("model")
    axes[0].set_ylabel("failure rate")

    sns.barplot(
        data=model_r5,
        x="model",
        y="r5_nonzero_rate",
        hue="model",
        palette="flare",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("r_5 > 0 Rate by Model")
    axes[1].set_xlabel("model")
    axes[1].set_ylabel("non-zero rate")

    fig.suptitle("r_5 Signal by Model: More Interpretable than Boxplot", fontsize=14)
    fig.savefig(FIG_DIR / "r5_model_signal_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return summary


def save_corr_heatmap(dedup: pd.DataFrame) -> None:
    corr_data = dedup[FEATURES + ["failure"]].copy()
    for col in RAW_FEATURES:
        corr_data[col] = np.log1p(corr_data[col].clip(lower=0))
    corr = corr_data.corr(numeric_only=True)
    corr.to_csv(RESULT_DIR / "smart_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("SMART Feature Correlation Heatmap")
    fig.savefig(FIG_DIR / "smart_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sample_for_distribution(dedup: pd.DataFrame, n_each: int = 8000) -> pd.DataFrame:
    samples = []
    for label, group in dedup.groupby("failure"):
        samples.append(group.sample(n=min(n_each, len(group)), random_state=42))
    return pd.concat(samples, ignore_index=True)


def save_violin_plot(dedup: pd.DataFrame) -> None:
    sample = sample_for_distribution(dedup)
    long_df = sample.melt(
        id_vars=["failure"],
        value_vars=FEATURES,
        var_name="feature",
        value_name="value",
    )
    long_df["log_value"] = np.log1p(long_df["value"].clip(lower=0))

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    sns.violinplot(
        data=long_df,
        x="feature",
        y="log_value",
        hue="failure",
        split=True,
        inner="quartile",
        palette={0: "#4C78A8", 1: "#F58518"},
        cut=0,
        ax=ax,
    )
    ax.set_title("Healthy vs Failed Disks: Multi-feature Distribution")
    ax.set_xlabel("SMART feature")
    ax.set_ylabel("log1p(value)")
    ax.legend(title="failure", labels=["healthy", "failed"])
    fig.savefig(FIG_DIR / "smart_multi_metric_violin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_plots_and_stats(dedup: pd.DataFrame) -> pd.DataFrame:
    model_stats = (
        dedup.groupby("model")
        .agg(
            disk_count=("disk_id", "count"),
            failed_disks=("failure", "sum"),
            failure_rate=("failure", "mean"),
            r_5_median=("r_5", "median"),
            r_9_median=("r_9", "median"),
            r_12_median=("r_12", "median"),
            r_175_median=("r_175", "median"),
        )
        .reset_index()
        .sort_values("failure_rate", ascending=False)
    )
    model_stats.to_csv(RESULT_DIR / "model_failure_feature_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    sns.barplot(
        data=model_stats,
        x="model",
        y="failure_rate",
        hue="model",
        palette="viridis",
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Failure rate by model")
    axes[0, 0].set_ylabel("failure rate")

    for ax, feature in zip(axes.ravel()[1:], ["r_5", "r_9", "r_12"]):
        temp = dedup[["model", feature]].copy()
        temp[feature] = np.log1p(temp[feature].clip(lower=0))
        sns.boxplot(data=temp, x="model", y=feature, showfliers=False, color="#72B7B2", ax=ax)
        ax.set_title(f"{feature} distribution by model")
        ax.set_ylabel(f"log1p({feature})")
    for ax in axes.ravel():
        ax.set_xlabel("model")
    fig.savefig(FIG_DIR / "model_failure_and_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return model_stats


def save_location_plots_and_stats(dedup: pd.DataFrame) -> dict[str, pd.DataFrame]:
    global_rate = float(dedup["failure"].mean())

    def summarize_group(col: str, min_n: int = 1) -> pd.DataFrame:
        stats = (
            dedup.groupby(col, dropna=False)
            .agg(
                sample_count=("failure", "size"),
                failed_count=("failure", "sum"),
                failure_rate=("failure", "mean"),
            )
            .reset_index()
        )
        stats["expected_failed_at_global_rate"] = stats["sample_count"] * global_rate
        stats["lift_vs_global"] = stats["failure_rate"] / global_rate
        stats = stats[stats["sample_count"] >= min_n].sort_values(
            ["failure_rate", "failed_count"], ascending=False
        )
        return stats

    app_stats = summarize_group("app")
    app_stats.to_csv(RESULT_DIR / "app_failure_summary.csv", index=False)

    slot_stats = summarize_group("slot_id_grouped")
    slot_stats.to_csv(RESULT_DIR / "slot_failure_summary.csv", index=False)

    rack_stats = summarize_group("rack_id", min_n=100)
    rack_stats.to_csv(RESULT_DIR / "rack_failure_hotspots.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    app_plot = app_stats.sort_values("failure_rate", ascending=False)
    sns.barplot(data=app_plot, x="app", y="failure_rate", hue="app", palette="crest", legend=False, ax=ax)
    ax.axhline(global_rate, color="#D62728", linestyle="--", linewidth=1.4, label="global rate")
    ax.set_title("Failure Rate by Application")
    ax.set_xlabel("application")
    ax.set_ylabel("failure rate")
    ax.legend()
    fig.savefig(FIG_DIR / "location_app_failure_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    slot_plot = slot_stats[slot_stats["sample_count"] >= 100].sort_values("failure_rate", ascending=False)
    sns.barplot(
        data=slot_plot,
        x="slot_id_grouped",
        y="failure_rate",
        hue="slot_id_grouped",
        palette="mako",
        legend=False,
        ax=ax,
    )
    ax.axhline(global_rate, color="#D62728", linestyle="--", linewidth=1.4, label="global rate")
    ax.set_title("Failure Rate by Slot")
    ax.set_xlabel("slot_id")
    ax.set_ylabel("failure rate")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.savefig(FIG_DIR / "location_slot_failure_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    rack_plot = rack_stats.head(15).copy()
    rack_plot["rack_id"] = rack_plot["rack_id"].astype(int).astype(str)
    sns.barplot(
        data=rack_plot,
        y="rack_id",
        x="failure_rate",
        hue="rack_id",
        palette="flare",
        legend=False,
        ax=ax,
    )
    ax.axvline(global_rate, color="#1F77B4", linestyle="--", linewidth=1.4, label="global rate")
    ax.set_title("Top Rack Failure Hotspots (sample_count >= 100)")
    ax.set_xlabel("failure rate")
    ax.set_ylabel("rack_id")
    ax.legend()
    fig.savefig(FIG_DIR / "location_rack_hotspots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "app_stats": app_stats,
        "slot_stats": slot_stats,
        "rack_stats": rack_stats,
    }


def indicator_summary(dedup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        healthy = dedup.loc[dedup["failure"] == 0, feature]
        failed = dedup.loc[dedup["failure"] == 1, feature]
        healthy_median = float(healthy.median())
        failed_median = float(failed.median())
        healthy_mean = float(healthy.mean())
        failed_mean = float(failed.mean())
        rows.append(
            {
                "feature": feature,
                "healthy_median": healthy_median,
                "failed_median": failed_median,
                "median_diff": failed_median - healthy_median,
                "healthy_mean": healthy_mean,
                "failed_mean": failed_mean,
                "mean_ratio_failed_to_healthy": failed_mean / healthy_mean
                if healthy_mean not in [0, np.nan] and healthy_mean != 0
                else np.nan,
                "missing_pct": float(dedup[feature].isna().mean() * 100),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "indicator_failure_comparison.csv", index=False)
    return summary


def evaluate_predictions(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    cm = confusion_matrix(y_test, y_pred)

    top_k = {}
    order = np.argsort(y_prob)[::-1]
    positives = int(y_test.sum())
    y_array = y_test.to_numpy()
    for k in [100, 500, 1000, 3000, 5000]:
        k_eff = min(k, len(y_array))
        hits = int(y_array[order[:k_eff]].sum())
        top_k[str(k)] = {
            "hits": hits,
            "precision": round(hits / k_eff, 4),
            "recall": round(hits / positives, 4) if positives else 0.0,
        }

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_failed": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall_failed": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_failed": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "average_precision": round(float(average_precision_score(y_test, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
        "top_k": top_k,
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["healthy", "failed"],
            zero_division=0,
            output_dict=True,
        ),
    }


def train_failure_model(dedup: pd.DataFrame) -> dict:
    model_df = dedup[FEATURES + CATEGORICAL_FEATURES + LOCATION_MODEL_FEATURES + ["failure"]].copy()
    X = model_df[FEATURES + CATEGORICAL_FEATURES + LOCATION_MODEL_FEATURES]
    y = model_df["failure"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ]
    )

    numeric_clf = Pipeline(
        steps=[
            ("preprocess", numeric_preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    numeric_clf.fit(X_train[FEATURES], y_train)

    numeric_pred = numeric_clf.predict(X_test[FEATURES])
    numeric_prob = numeric_clf.predict_proba(X_test[FEATURES])[:, 1]
    numeric_result = evaluate_predictions(y_test, numeric_pred, numeric_prob)

    def categorical_logistic_pipeline(cat_features: list[str]) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    FEATURES,
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    cat_features,
                ),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    model_clf = categorical_logistic_pipeline(CATEGORICAL_FEATURES)
    model_clf.fit(X_train[FEATURES + CATEGORICAL_FEATURES], y_train)

    model_pred = model_clf.predict(X_test[FEATURES + CATEGORICAL_FEATURES])
    model_prob = model_clf.predict_proba(X_test[FEATURES + CATEGORICAL_FEATURES])[:, 1]
    model_result = evaluate_predictions(y_test, model_pred, model_prob)

    location_features = CATEGORICAL_FEATURES + LOCATION_MODEL_FEATURES
    location_clf = categorical_logistic_pipeline(location_features)
    location_clf.fit(X_train[FEATURES + location_features], y_train)

    location_pred = location_clf.predict(X_test[FEATURES + location_features])
    location_prob = location_clf.predict_proba(X_test[FEATURES + location_features])[:, 1]
    location_result = evaluate_predictions(y_test, location_pred, location_prob)

    cat_names = list(
        location_clf.named_steps["preprocess"]
        .named_transformers_["cat"]
        .get_feature_names_out(location_features)
    )
    feature_names = FEATURES + cat_names
    coefs = location_clf.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(RESULT_DIR / "logistic_regression_coefficients.csv", index=False)

    cm = np.array(location_result["confusion_matrix"])
    pd.DataFrame(cm, index=["actual_healthy", "actual_failed"], columns=["pred_healthy", "pred_failed"]).to_csv(
        RESULT_DIR / "model_confusion_matrix.csv"
    )

    result = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_features": FEATURES,
        "model_aware_features": FEATURES + CATEGORICAL_FEATURES,
        "location_aware_features": FEATURES + CATEGORICAL_FEATURES + LOCATION_MODEL_FEATURES,
        "excluded_location_features": ["rack_id", "node_id"],
        "class_weight": "balanced",
        "positive_rate_test": round(float(y_test.mean()), 6),
        "baseline_numeric_logistic": numeric_result,
        "model_aware_logistic": model_result,
        "location_aware_logistic": location_result,
        "top_coefficients": coef_df.head(15).to_dict(orient="records"),
    }
    with open(RESULT_DIR / "failure_prediction_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def write_run_summary(
    summary: dict,
    indicators: pd.DataFrame,
    smart_interpretation: pd.DataFrame,
    model_stats: pd.DataFrame,
    location_stats: dict[str, pd.DataFrame],
    model_result: dict,
) -> None:
    def clean_json(value):
        if isinstance(value, dict):
            return {k: clean_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean_json(v) for v in value]
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value

    payload = {
        "data_summary": summary,
        "indicator_comparison": indicators.to_dict(orient="records"),
        "smart_interpretation": smart_interpretation.to_dict(orient="records"),
        "top_failure_models": model_stats.head(5).to_dict(orient="records"),
        "top_failure_apps": location_stats["app_stats"].head(5).to_dict(orient="records"),
        "top_failure_slots": location_stats["slot_stats"].head(5).to_dict(orient="records"),
        "top_failure_racks": location_stats["rack_stats"].head(10).to_dict(orient="records"),
        "model_result": model_result,
        "outputs": {
            "dedup_csv": str(DEDUP_PATH),
            "figures": [
                "figures/smart_other_boxplots.png",
                "figures/smart_indicator_interpretation.png",
                "figures/r5_model_signal_summary.png",
                "figures/smart_correlation_heatmap.png",
                "figures/smart_multi_metric_violin.png",
                "figures/model_failure_and_metrics.png",
                "figures/location_app_failure_rate.png",
                "figures/location_slot_failure_rate.png",
                "figures/location_rack_hotspots.png",
            ],
            "results": [
                "results/indicator_failure_comparison.csv",
                "results/smart_feature_interpretation_summary.csv",
                "results/r5_model_signal_summary.csv",
                "results/model_failure_feature_summary.csv",
                "results/app_failure_summary.csv",
                "results/slot_failure_summary.csv",
                "results/rack_failure_hotspots.csv",
                "results/failure_prediction_metrics.json",
                "results/logistic_regression_coefficients.csv",
                "results/model_confusion_matrix.csv",
            ],
        },
    }
    with open(RESULT_DIR / "b_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(clean_json(payload), f, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    ensure_dirs()
    _, dedup, summary = load_and_dedup()
    save_smart_boxplots(dedup)
    smart_interpretation = save_interpretable_smart_plots(dedup)
    save_corr_heatmap(dedup)
    save_violin_plot(dedup)
    model_stats = save_model_plots_and_stats(dedup)
    location_stats = save_location_plots_and_stats(dedup)
    indicators = indicator_summary(dedup)
    model_result = train_failure_model(dedup)
    write_run_summary(summary, indicators, smart_interpretation, model_stats, location_stats, model_result)

    print("B analysis completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    compact = {
        "baseline_numeric_logistic": {
            k: model_result["baseline_numeric_logistic"][k]
            for k in ["accuracy", "precision_failed", "recall_failed", "f1_failed", "roc_auc", "average_precision"]
        },
        "model_aware_logistic": {
            k: model_result["model_aware_logistic"][k]
            for k in ["accuracy", "precision_failed", "recall_failed", "f1_failed", "roc_auc", "average_precision"]
        },
        "location_aware_logistic": {
            k: model_result["location_aware_logistic"][k]
            for k in ["accuracy", "precision_failed", "recall_failed", "f1_failed", "roc_auc", "average_precision"]
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
