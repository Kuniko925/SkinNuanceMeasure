from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


# ============================================================
# Input files
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results_eval"

FILES = {
    ("BCN", "3G"):
        RESULTS_DIR / "group_summary_BCN_test_seed_level_skin_tone_group_id.csv",

    ("BCN", "6G"):
        RESULTS_DIR / "group_summary_BCN_test_seed_level_ita_skin_type.csv",

    ("HAM", "3G"):
        RESULTS_DIR / "group_summary_HAM_test_seed_level_skin_tone_group_id.csv",

    ("HAM", "6G"):
        RESULTS_DIR / "group_summary_HAM_test_seed_level_ita_skin_type.csv",
}


METHODS = [
    "baseline",
    "reweight",
    "individual",
]

METRICS = [
    "Accuracy",
    "Macro-F1",
    "Accuracy gap",
    "TPR gap",
    "FPR gap",
    "Equalized-Odds gap",
]


# ============================================================
# Helper: calculate metrics for one
# dataset × grouping × model × method × seed
# ============================================================

def calculate_metrics(sub):

    # --------------------------------------------------------
    # Overall Accuracy and Macro-F1
    #
    # Aggregate confusion matrix across skin-tone groups.
    # --------------------------------------------------------

    TP = sub["TP"].sum()
    FN = sub["FN"].sum()
    FP = sub["FP"].sum()
    TN = sub["TN"].sum()

    total = TP + FN + FP + TN

    accuracy = (
        (TP + TN) / total
        if total > 0
        else np.nan
    )

    # F1 for malignant class
    denom_pos = 2 * TP + FP + FN
    f1_pos = (
        2 * TP / denom_pos
        if denom_pos > 0
        else np.nan
    )

    # F1 for benign class
    denom_neg = 2 * TN + FP + FN
    f1_neg = (
        2 * TN / denom_neg
        if denom_neg > 0
        else np.nan
    )

    macro_f1 = np.nanmean([f1_pos, f1_neg])


    # --------------------------------------------------------
    # Accuracy gap
    # --------------------------------------------------------

    accuracy_gap = (
        sub["accuracy"].max()
        - sub["accuracy"].min()
    )


    # --------------------------------------------------------
    # TPR gap
    # --------------------------------------------------------

    tpr = sub["TPR_malignant"].dropna()

    tpr_gap = (
        tpr.max() - tpr.min()
        if len(tpr) >= 2
        else np.nan
    )


    # --------------------------------------------------------
    # FPR gap
    # --------------------------------------------------------

    fpr = sub["FPR_malignant"].dropna()

    fpr_gap = (
        fpr.max() - fpr.min()
        if len(fpr) >= 2
        else np.nan
    )


    # --------------------------------------------------------
    # Equalized-Odds gap
    #
    # Important:
    # use only groups where BOTH TPR and FPR are defined.
    # This reproduces the convention used in the summary CSVs.
    # --------------------------------------------------------

    valid = (
        sub["TPR_malignant"].notna()
        & sub["FPR_malignant"].notna()
    )

    eo_sub = sub.loc[valid]

    if len(eo_sub) >= 2:

        eo_tpr_gap = (
            eo_sub["TPR_malignant"].max()
            - eo_sub["TPR_malignant"].min()
        )

        eo_fpr_gap = (
            eo_sub["FPR_malignant"].max()
            - eo_sub["FPR_malignant"].min()
        )

        eodds_gap = max(
            eo_tpr_gap,
            eo_fpr_gap
        )

    else:
        eodds_gap = np.nan


    return {
        "Accuracy": accuracy,
        "Macro-F1": macro_f1,
        "Accuracy gap": accuracy_gap,
        "TPR gap": tpr_gap,
        "FPR gap": fpr_gap,
        "Equalized-Odds gap": eodds_gap,
    }


# ============================================================
# Step 1:
# Calculate metrics at seed level
# ============================================================

records = []

for (dataset, grouping), filepath in FILES.items():

    df = pd.read_csv(filepath)

    grouped = df.groupby(
        ["model_name", "train_mode", "seed"],
        sort=False
    )

    for (model, method, seed), sub in grouped:

        if method not in METHODS:
            continue

        values = calculate_metrics(sub)

        for metric, value in values.items():

            records.append({
                "dataset": dataset,
                "grouping": grouping,
                "model": model,
                "method": method,
                "seed": seed,
                "metric": metric,
                "value": value,
            })


metric_df = pd.DataFrame(records)


# ============================================================
# Step 2:
# Average dataset × model WITHIN each seed
#
# 2 datasets × 4 models = 8 conditions per seed
# ============================================================

seed_means = (
    metric_df
    .groupby(
        [
            "grouping",
            "method",
            "metric",
            "seed",
        ],
        as_index=False
    )["value"]
    .mean()
)


# ============================================================
# Optional sanity check:
# each grouping/method/metric should have 5 seeds
# ============================================================

seed_counts = (
    seed_means
    .groupby(
        ["grouping", "method", "metric"]
    )["seed"]
    .nunique()
)

print("\nSeed counts:")
print(seed_counts)


# ============================================================
# Step 3:
# Align baseline / reweight / individual by SAME seed
# ============================================================

paired = (
    seed_means
    .pivot(
        index=[
            "grouping",
            "metric",
            "seed",
        ],
        columns="method",
        values="value"
    )
    .reset_index()
)


# ============================================================
# Step 4:
# Calculate paired differences
#
# Reweight - Baseline
# Individual - Baseline
# ============================================================

paired["reweight_minus_baseline"] = (
    paired["reweight"]
    - paired["baseline"]
)

paired["individual_minus_baseline"] = (
    paired["individual"]
    - paired["baseline"]
)


# ============================================================
# 95% CI using Student's t distribution
# ============================================================

def mean_ci(values, confidence=0.95):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    mean = values.mean()

    if n == 1:
        return mean, np.nan, np.nan

    sd = values.std(ddof=1)

    alpha = 1 - confidence

    critical = t.ppf(
        1 - alpha / 2,
        df=n - 1
    )

    margin = (
        critical
        * sd
        / np.sqrt(n)
    )

    return (
        mean,
        mean - margin,
        mean + margin,
    )


# ============================================================
# Step 5:
# Summarize paired differences
# ============================================================

summary_records = []

for grouping in ["3G", "6G"]:

    for metric in METRICS:

        for comparison, column in [
            (
                "Reweight−Baseline",
                "reweight_minus_baseline"
            ),
            (
                "Individual−Baseline",
                "individual_minus_baseline"
            ),
        ]:

            values = paired.loc[
                (paired["grouping"] == grouping)
                & (paired["metric"] == metric),
                column
            ]

            mean, lower, upper = mean_ci(values)

            summary_records.append({
                "grouping": grouping,
                "metric": metric,
                "comparison": comparison,

                # percentage points
                "mean": mean * 100,
                "ci_lower": lower * 100,
                "ci_upper": upper * 100,
            })


summary = pd.DataFrame(summary_records)


# ============================================================
# Formatting:
# -0.32 [-0.64, -0.00]
# ============================================================

def format_ci(row):

    return (
        f"{row['mean']:.2f} "
        f"[{row['ci_lower']:.2f}, "
        f"{row['ci_upper']:.2f}]"
    )


summary["formatted"] = summary.apply(
    format_ci,
    axis=1
)


# ============================================================
# Step 6:
# Build publication table
# ============================================================

rows = []

for metric in METRICS:

    row = {
        "Metric": metric
    }

    for grouping in ["3G", "6G"]:

        for comparison in [
            "Reweight−Baseline",
            "Individual−Baseline",
        ]:

            value = summary.loc[
                (summary["metric"] == metric)
                & (summary["grouping"] == grouping)
                & (
                    summary["comparison"]
                    == comparison
                ),
                "formatted"
            ].iloc[0]

            column_name = (
                f"{comparison} {grouping}"
            )

            row[column_name] = value

    rows.append(row)


table = pd.DataFrame(rows)


# ============================================================
# Column order
# ============================================================

table = table[
    [
        "Metric",
        "Reweight−Baseline 3G",
        "Individual−Baseline 3G",
        "Reweight−Baseline 6G",
        "Individual−Baseline 6G",
    ]
]


# ============================================================
# Display
# ============================================================

print("\nFinal publication table:\n")
print(
    table.to_string(index=False)
)


# ============================================================
# Optional:
# inspect seed-level paired differences
# ============================================================

debug = paired.copy()

for column in [
    "reweight_minus_baseline",
    "individual_minus_baseline",
]:
    debug[column + "_pp"] = (
        debug[column] * 100
    )

print("\nSeed-level paired differences:\n")

print(
    debug[
        [
            "grouping",
            "metric",
            "seed",
            "reweight_minus_baseline_pp",
            "individual_minus_baseline_pp",
        ]
    ].to_string(index=False)
)


# ============================================================
# Save CSV
# ============================================================

output_path = RESULTS_DIR / "baseline_comparison_performance_fairness_95CI.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

table.to_csv(
    output_path,
    index=False
)


print("\nSaved:")
print(f"  {output_path}")
