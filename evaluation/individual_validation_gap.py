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


METRICS = [
    "Accuracy gap",
    "TPR gap",
    "FPR gap",
    "Equalized-Odds gap",
]


# ============================================================
# Step 1:
# Calculate gap for each
#
# dataset × grouping × model × method × seed
# ============================================================

records = []

for (dataset, grouping), filepath in FILES.items():

    df = pd.read_csv(filepath)

    grouped = df.groupby(
        ["model_name", "train_mode", "seed"],
        sort=False
    )

    for (model, method, seed), sub in grouped:

        if method not in ["reweight", "individual"]:
            continue

        # ----------------------------------------
        # Accuracy gap
        # ----------------------------------------
        accuracy_gap = (
            sub["accuracy"].max()
            - sub["accuracy"].min()
        )

        # ----------------------------------------
        # TPR gap
        # ----------------------------------------
        tpr = sub["TPR_malignant"].dropna()

        tpr_gap = (
            tpr.max() - tpr.min()
            if len(tpr) >= 2
            else np.nan
        )

        # ----------------------------------------
        # FPR gap
        # ----------------------------------------
        fpr = sub["FPR_malignant"].dropna()

        fpr_gap = (
            fpr.max() - fpr.min()
            if len(fpr) >= 2
            else np.nan
        )

        # ----------------------------------------
        # Equalized-Odds gap
        #
        # Use only groups where BOTH TPR and FPR
        # are defined.
        # ----------------------------------------
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

        metric_values = {
            "Accuracy gap": accuracy_gap,
            "TPR gap": tpr_gap,
            "FPR gap": fpr_gap,
            "Equalized-Odds gap": eodds_gap,
        }

        for metric, value in metric_values.items():

            records.append({
                "dataset": dataset,
                "grouping": grouping,
                "model": model,
                "method": method,
                "seed": seed,
                "metric": metric,
                "value": value,
            })


gap_df = pd.DataFrame(records)


# ============================================================
# Step 2:
# Average over dataset × model WITHIN each seed
#
# This gives one value per:
# grouping × method × metric × seed
# ============================================================

seed_means = (
    gap_df
    .groupby(
        [
            "grouping",
            "method",
            "metric",
            "seed"
        ],
        as_index=False
    )["value"]
    .mean()
)


# ============================================================
# Step 3:
# Reshape so Reweight and Individual are aligned
# for the SAME seed
# ============================================================

paired = (
    seed_means
    .pivot(
        index=[
            "grouping",
            "metric",
            "seed"
        ],
        columns="method",
        values="value"
    )
    .reset_index()
)


# ============================================================
# Step 4:
# Paired difference
#
# Individual - Reweight
#
# For fairness gaps:
#   negative = Individual is better
# ============================================================

paired["difference"] = (
    paired["individual"]
    - paired["reweight"]
)


# ============================================================
# Step 5:
# Mean difference + t-based 95% CI
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
        mean + margin
    )


summary_records = []

for grouping in ["3G", "6G"]:

    for metric in METRICS:

        values = paired.loc[
            (paired["grouping"] == grouping)
            & (paired["metric"] == metric),
            "difference"
        ]

        mean, lower, upper = mean_ci(values)

        summary_records.append({
            "grouping": grouping,
            "metric": metric,

            # percentage points
            "mean_difference": mean * 100,
            "ci_lower": lower * 100,
            "ci_upper": upper * 100,
        })


summary = pd.DataFrame(summary_records)


# ============================================================
# Step 6:
# Formatting
#
# Example:
# -1.71 [-2.54, -0.88]
# ============================================================

def fmt(row):

    return (
        f"{row['mean_difference']:.2f} "
        f"[{row['ci_lower']:.2f}, "
        f"{row['ci_upper']:.2f}]"
    )


summary["formatted"] = summary.apply(
    fmt,
    axis=1
)


# ============================================================
# Step 7:
# Publication table
# ============================================================

rows = []

for metric in METRICS:

    row = {
        "Metric": metric
    }

    for grouping in ["3G", "6G"]:

        value = summary.loc[
            (summary["metric"] == metric)
            & (summary["grouping"] == grouping),
            "formatted"
        ].iloc[0]

        if grouping == "3G":
            row["3-group difference"] = value
        else:
            row["6-group difference"] = value

    rows.append(row)


table = pd.DataFrame(rows)


# ============================================================
# Print
# ============================================================

print("\nPaired Individual - Reweight differences:\n")
print(table.to_string(index=False))


# ============================================================
# Optional:
# show seed-level differences for verification
# ============================================================

print("\nSeed-level paired differences:\n")

debug_table = paired.copy()

debug_table["difference_pp"] = (
    debug_table["difference"] * 100
)

print(
    debug_table[
        [
            "grouping",
            "metric",
            "seed",
            "reweight",
            "individual",
            "difference_pp"
        ]
    ].to_string(index=False)
)


# ============================================================
# Save CSV
# ============================================================

output_path = RESULTS_DIR / "paired_individual_vs_reweight_95CI.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

table.to_csv(
    output_path,
    index=False
)


print("\nSaved:")
print(f"  {output_path}")
