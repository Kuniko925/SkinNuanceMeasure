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


METHODS = ["reweight", "individual"]

METRICS = [
    "Accuracy gap",
    "TPR gap",
    "FPR gap",
    "Equalized-Odds gap",
]


# ============================================================
# Calculate fairness gaps for each:
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

        if method not in METHODS:
            continue

        # ----------------------------------------------------
        # Accuracy gap
        # max(group accuracy) - min(group accuracy)
        # ----------------------------------------------------

        accuracy_gap = (
            sub["accuracy"].max()
            - sub["accuracy"].min()
        )

        # ----------------------------------------------------
        # TPR gap
        #
        # Groups for which TPR cannot be computed
        # (e.g. zero malignant samples) are ignored.
        # ----------------------------------------------------

        tpr = sub["TPR_malignant"].dropna()

        if len(tpr) >= 2:
            tpr_gap = tpr.max() - tpr.min()
        else:
            tpr_gap = np.nan

        # ----------------------------------------------------
        # FPR gap
        # ----------------------------------------------------

        fpr = sub["FPR_malignant"].dropna()

        if len(fpr) >= 2:
            fpr_gap = fpr.max() - fpr.min()
        else:
            fpr_gap = np.nan

        # ----------------------------------------------------
        # Equalized-Odds gap
        #
        # IMPORTANT:
        # For EO, use only groups for which BOTH
        # TPR and FPR are defined.
        #
        # This matters especially for HAM 6G,
        # where some groups contain zero malignant samples.
        # ----------------------------------------------------

        valid_eo = (
            sub["TPR_malignant"].notna()
            & sub["FPR_malignant"].notna()
        )

        eo_sub = sub.loc[valid_eo]

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

        values = {
            "Accuracy gap": accuracy_gap,
            "TPR gap": tpr_gap,
            "FPR gap": fpr_gap,
            "Equalized-Odds gap": eodds_gap,
        }

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


gap_df = pd.DataFrame(records)


# ============================================================
# For each seed:
#
# Average over:
#   2 datasets × 4 models = 8 conditions
#
# This leaves 5 observations per:
#   grouping × method × metric
# ============================================================

seed_means = (
    gap_df
    .groupby(
        ["grouping", "method", "metric", "seed"],
        as_index=False
    )["value"]
    .mean()
)


# ============================================================
# 95% CI using Student's t distribution
#
# mean ± t_(0.975, n-1) * SD / sqrt(n)
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

    margin = critical * sd / np.sqrt(n)

    return (
        mean,
        mean - margin,
        mean + margin
    )


# ============================================================
# Build summary
# ============================================================

summary_records = []

for grouping in ["3G", "6G"]:

    for method in METHODS:

        for metric in METRICS:

            values = seed_means.loc[
                (seed_means["grouping"] == grouping)
                & (seed_means["method"] == method)
                & (seed_means["metric"] == metric),
                "value"
            ]

            mean, lower, upper = mean_ci(values)

            summary_records.append({
                "grouping": grouping,
                "method": method,
                "metric": metric,

                # convert to percentage points
                "mean": mean * 100,
                "ci_lower": lower * 100,
                "ci_upper": upper * 100,
            })


summary = pd.DataFrame(summary_records)


# ============================================================
# Formatting
#
# Example:
# 11.50 [11.07, 11.94]
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
# Create publication table
# ============================================================

rows = []

for metric in METRICS:

    row = {
        "Fairness metric": metric
    }

    for grouping in ["3G", "6G"]:

        for method in METHODS:

            value = summary.loc[
                (summary["metric"] == metric)
                & (summary["grouping"] == grouping)
                & (summary["method"] == method),
                "formatted"
            ].iloc[0]

            if method == "reweight":
                column = f"Reweight {grouping}"

            else:
                column = f"Individual {grouping}"

            row[column] = value

    rows.append(row)


table = pd.DataFrame(rows)


# Reorder columns
table = table[
    [
        "Fairness metric",
        "Reweight 3G",
        "Individual 3G",
        "Reweight 6G",
        "Individual 6G",
    ]
]


# ============================================================
# Print
# ============================================================

print("\nPublication table:\n")
print(table.to_string(index=False))


# ============================================================
# Save CSV
# ============================================================

output_path = RESULTS_DIR / "fairness_reweight_vs_individual_95CI.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

table.to_csv(
    output_path,
    index=False
)


print("\nSaved:")
print(f"  {output_path}")
