from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_REGISTRY

import pandas as pd
import numpy as np

# ============================================================
# Input files
# skin_tone_group_id = 3 groups
# ita_skin_type      = 6 groups
# ============================================================

INPUT_DIR = PROJECT_ROOT / "results_eval"

FILES = {
    ("BCN", "3-group"):
        INPUT_DIR / "fairness_summary_BCN_test_wide_skin_tone_group_id.csv",

    ("BCN", "6-group"):
        INPUT_DIR / "fairness_summary_BCN_test_wide_ita_skin_type.csv",

    ("HAM", "3-group"):
        INPUT_DIR / "fairness_summary_HAM_test_wide_skin_tone_group_id.csv",

    ("HAM", "6-group"):
        INPUT_DIR / "fairness_summary_HAM_test_wide_ita_skin_type.csv",
}


# ============================================================
# Metrics to include
# ============================================================

METRICS = {
    "Accuracy gap": "AccuracyGap_mean",
    "F1 gap": "F1Gap_mean",
    "TPR gap": "TPRGap_mean",
    "FPR gap": "FPRGap_mean",
    "Equalized-Odds gap": "Eodds_mean",
}


# ============================================================
# Helper functions
# ============================================================

def load_fairness_summary(path):
    """
    Read the wide fairness-summary CSV.

    The CSV has two header rows:
        level 0: baseline / reweight / individual
        level 1: Accuracy_mean, AccuracyGap_mean, ...
    """
    df = pd.read_csv(path, header=[0, 1])

    # The first column is ('Dataset', 'model_name')
    df = df.set_index(("Dataset", "model_name"))

    return df


def format_gap(v3, v6, decimals=1):
    """
    Convert proportions to percentage points and format as:
        8.1→11.2 (+3.1)
    """
    v3 = v3 * 100
    v6 = v6 * 100
    delta = v6 - v3

    return (
        f"{v3:.{decimals}f}"
        f"→{v6:.{decimals}f} "
        f"({delta:+.{decimals}f})"
    )


# ============================================================
# Build table
# ============================================================

table_rows = []

# Keep raw numbers separately for computing the overall mean
raw_values = {
    metric_label: {
        "3-group": [],
        "6-group": [],
    }
    for metric_label in METRICS
}


for dataset in ["BCN", "HAM"]:

    df3 = load_fairness_summary(FILES[(dataset, "3-group")])
    df6 = load_fairness_summary(FILES[(dataset, "6-group")])

    # Ensure same models are available
    common_models = [
        model for model in MODEL_REGISTRY
        if model in df3.index and model in df6.index
    ]

    for model in common_models:

        row = {
            "Dataset": dataset,
            "Model": model,
        }

        for display_name, csv_metric in METRICS.items():

            # baseline only
            value_3 = float(
                df3.loc[
                    model,
                    ("baseline", csv_metric)
                ]
            )

            value_6 = float(
                df6.loc[
                    model,
                    ("baseline", csv_metric)
                ]
            )

            raw_values[display_name]["3-group"].append(value_3)
            raw_values[display_name]["6-group"].append(value_6)

            row[display_name] = format_gap(
                value_3,
                value_6,
                decimals=1
            )

        table_rows.append(row)


# ============================================================
# Add overall mean row
# ============================================================

mean_row = {
    "Dataset": "Mean",
    "Model": "—",
}

for display_name in METRICS:

    mean_3 = np.mean(
        raw_values[display_name]["3-group"]
    )

    mean_6 = np.mean(
        raw_values[display_name]["6-group"]
    )

    mean_row[display_name] = format_gap(
        mean_3,
        mean_6,
        decimals=1
    )

table_rows.append(mean_row)


# ============================================================
# Final DataFrame
# ============================================================

table = pd.DataFrame(table_rows)

print("\n=== Baseline: 3-group vs 6-group fairness gaps ===\n")
print(table.to_string(index=False))


# ============================================================
# Save as CSV
# ============================================================

output_path = PROJECT_ROOT / "results_eval" / "table_baseline_3groups_vs_6groups.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

table.to_csv(
    output_path,
    index=False
)


print("\nSaved:")
print(f"  {output_path}")
