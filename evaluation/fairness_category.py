from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import TRAIN_MODES, DATASETS, SEEDS, MODEL_REGISTRY

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# Binary task definition
# 1 = malignant (positive), 0 = benign (negative)
POSITIVE_CLASS = 1

# Evaluate both categorical skin-tone definitions in one run.
# ita_skin_type: 6 groups
# skin_tone_group_id: 3 groups
GROUP_COLS = [
    "ita_skin_type",
    "skin_tone_group_id",
]


def _binary_rates(y_true, y_pred, positive_class=POSITIVE_CLASS):
    """Return malignant-positive confusion counts and rates."""
    yt = np.asarray(y_true) == positive_class
    yp = np.asarray(y_pred) == positive_class

    tp = int(np.sum(yt & yp))
    fn = int(np.sum(yt & (~yp)))
    fp = int(np.sum((~yt) & yp))
    tn = int(np.sum((~yt) & (~yp)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "TPR_malignant": tpr,
        "FPR_malignant": fpr,
        "FNR_malignant": fnr,
        "specificity": specificity,
    }


def _nan_range(values):
    """max - min while ignoring NaN; return NaN if no finite value exists."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(values.max() - values.min())


def fairness_summary_binary(
    df,
    label_col="label",
    pred_col="pred",
    group_col="ita_skin_type",
    positive_class=POSITIVE_CLASS,
):
    """
    Fairness summary for binary benign/malignant classification.

    The positive class is malignant only. Therefore:
      - TPR_malignant = malignant sensitivity / recall
      - FPR_malignant = benign samples incorrectly predicted malignant
      - Equal Opportunity Difference = max group TPR - min group TPR
      - Equalized Odds Difference = maximum pairwise disparity in TPR or FPR
    """
    required = {label_col, pred_col, group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    y_true = df[label_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    groups = np.sort(df[group_col].dropna().unique())

    # This analysis assumes a binary task.
    labels_seen = set(pd.unique(pd.concat([
        pd.Series(y_true),
        pd.Series(y_pred),
    ], ignore_index=True)))
    if positive_class not in labels_seen:
        raise ValueError(
            f"positive_class={positive_class} was not found in label/pred values: "
            f"{sorted(labels_seen)}"
        )

    overall_accuracy = float(accuracy_score(y_true, y_pred))
    overall_omr = 1.0 - overall_accuracy
    overall_f1_macro = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    group_rows = []

    for grp in groups:
        mask = df[group_col].to_numpy() == grp
        yt = y_true[mask]
        yp = y_pred[mask]

        rates = _binary_rates(yt, yp, positive_class=positive_class)
        acc = float(accuracy_score(yt, yp))
        f1_macro = float(
            f1_score(yt, yp, average="macro", zero_division=0)
        )

        n_positive = int(np.sum(yt == positive_class))
        n_negative = int(np.sum(yt != positive_class))
        n_samples = int(len(yt))

        group_rows.append({
            "group": grp,
            "n_samples": n_samples,
            "n_malignant": n_positive,
            "n_benign": n_negative,
            "malignant_prevalence": (
                n_positive / n_samples if n_samples > 0 else np.nan
            ),
            "accuracy": acc,
            "group_misclassification_rate": 1.0 - acc,
            # Keep the old column name for compatibility; this is macro-F1.
            "f1": f1_macro,
            **rates,
        })

    group_df = pd.DataFrame(group_rows).sort_values("group").reset_index(drop=True)

    accuracy_gap = _nan_range(group_df["accuracy"])
    f1_gap = _nan_range(group_df["f1"])
    tpr_gap = _nan_range(group_df["TPR_malignant"])
    fpr_gap = _nan_range(group_df["FPR_malignant"])
    omr_gap = _nan_range(group_df["group_misclassification_rate"])

    # Equal Opportunity: malignant sensitivity disparity across groups.
    equal_opportunity_diff = tpr_gap

    # Equalized Odds: maximum pairwise difference in either malignant TPR or FPR.
    equalized_odds_diff = np.nan
    max_pair_gap = -np.inf

    for i in range(len(group_df)):
        for j in range(i + 1, len(group_df)):
            row_i = group_df.iloc[i]
            row_j = group_df.iloc[j]

            vals = [
                row_i["TPR_malignant"],
                row_j["TPR_malignant"],
                row_i["FPR_malignant"],
                row_j["FPR_malignant"],
            ]
            if not all(np.isfinite(v) for v in vals):
                continue

            pair_gap = max(
                abs(row_i["TPR_malignant"] - row_j["TPR_malignant"]),
                abs(row_i["FPR_malignant"] - row_j["FPR_malignant"]),
            )
            max_pair_gap = max(max_pair_gap, pair_gap)

    if np.isfinite(max_pair_gap):
        equalized_odds_diff = float(max_pair_gap)

    return {
        "overall_accuracy": overall_accuracy,
        "overall_misclassification_rate": overall_omr,
        "overall_f1_macro": overall_f1_macro,
        "accuracy_gap": accuracy_gap,
        "f1_gap": f1_gap,
        "TPR_malignant_gap": tpr_gap,
        "FPR_malignant_gap": fpr_gap,
        "group_misclassification_rate_gap": omr_gap,
        "equal_opportunity_diff": equal_opportunity_diff,
        "equalized_odds_diff": equalized_odds_diff,
        "group_metrics": group_df,
    }


def run_one_grouping(group_col):
    train_modes = TRAIN_MODES
    db_names = DATASETS
    model_names = list(MODEL_REGISTRY.keys())
    eval_types = ["test"]

    output_dir = PROJECT_ROOT / "results_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    for db_name in db_names:
        for eval_type in eval_types:
            records = []
            group_seed_records = []

            for model_name in model_names:
                row = {
                    ("Dataset", "model_name"): model_name,
                }

                for train_mode in train_modes:
                    seed_metrics = []

                    for seed in SEEDS:
                        filepath = (
                            PROJECT_ROOT
                            / f"results/{db_name}_{train_mode}_{model_name}_seed{seed}_{eval_type}.csv"
                        )

                        if not filepath.exists():
                            print(f"not found: {filepath}")
                            continue

                        df = pd.read_csv(filepath)

                        if group_col not in df.columns:
                            print(f"missing column '{group_col}': {filepath}")
                            continue

                        df = df.dropna(subset=[group_col])

                        if len(df) == 0:
                            print(f"no valid rows for '{group_col}': {filepath}")
                            continue

                        res = fairness_summary_binary(
                            df,
                            group_col=group_col,
                            positive_class=POSITIVE_CLASS,
                        )

                        # Seed-level fairness gaps for this model/train mode.
                        seed_metrics.append({
                            "seed": seed,
                            "Accuracy": res["overall_accuracy"],
                            "OMR": res["overall_misclassification_rate"],
                            "F1_macro": res["overall_f1_macro"],
                            "AccuracyGap": res["accuracy_gap"],
                            "F1Gap": res["f1_gap"],
                            "TPRGap": res["TPR_malignant_gap"],
                            "FPRGap": res["FPR_malignant_gap"],
                            "Eopp": res["equal_opportunity_diff"],
                            "Eodds": res["equalized_odds_diff"],
                        })

                        tmp_df = res["group_metrics"].copy()
                        tmp_df["dataset"] = db_name
                        tmp_df["model_name"] = model_name
                        tmp_df["train_mode"] = train_mode
                        tmp_df["seed"] = seed
                        tmp_df["group_definition"] = group_col

                        group_seed_records.append(tmp_df)

                    seed_df = pd.DataFrame(seed_metrics)
                    if len(seed_df) == 0:
                        continue

                    label = train_mode
                    metrics_for_wide = [
                        "Accuracy",
                        "OMR",
                        "F1_macro",
                        "AccuracyGap",
                        "F1Gap",
                        "TPRGap",
                        "FPRGap",
                        "Eopp",
                        "Eodds",
                    ]

                    for metric in metrics_for_wide:
                        row[(label, f"{metric}_mean")] = seed_df[metric].mean()
                        row[(label, f"{metric}_std")] = seed_df[metric].std(ddof=1)

                records.append(row)

            # ==================================================
            # Fairness summary: model x train_mode, averaged over seeds
            # ==================================================
            if records:
                summary_df = pd.DataFrame(records)
                summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns)

                output_path = (
                    output_dir
                    / f"fairness_summary_{db_name}_{eval_type}_wide_{group_col}.csv"
                )

                summary_df.to_csv(
                    output_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(f"saved: {output_path}")

            # ==================================================
            # Group metrics: seed-level and seed-aggregated
            # ==================================================
            if group_seed_records:
                group_seed_df = pd.concat(
                    group_seed_records,
                    ignore_index=True,
                )

                seed_group_path = (
                    output_dir
                    / f"group_summary_{db_name}_{eval_type}_seed_level_{group_col}.csv"
                )
                group_seed_df.to_csv(
                    seed_group_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(f"saved: {seed_group_path}")

                metrics = [
                    "accuracy",
                    "group_misclassification_rate",
                    "f1",
                    "malignant_prevalence",
                    "TPR_malignant",
                    "FPR_malignant",
                    "FNR_malignant",
                    "specificity",
                ]

                agg_dict = {}
                for metric in metrics:
                    agg_dict[f"{metric}_mean"] = (metric, "mean")
                    agg_dict[f"{metric}_std"] = (metric, "std")

                group_summary_df = (
                    group_seed_df
                    .groupby(
                        [
                            "dataset",
                            "model_name",
                            "train_mode",
                            "group_definition",
                            "group",
                        ],
                        as_index=False,
                    )
                    .agg(
                        n_seeds=("seed", "nunique"),
                        n_samples=("n_samples", "first"),
                        n_malignant=("n_malignant", "first"),
                        n_benign=("n_benign", "first"),
                        **agg_dict,
                    )
                )

                performance_csv_path = (
                    output_dir
                    / f"group_summary_{db_name}_{eval_type}_all_modes_{group_col}.csv"
                )
                group_summary_df.to_csv(
                    performance_csv_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(f"saved: {performance_csv_path}")


def main():
    for group_col in GROUP_COLS:
        print(f"\n=== group definition: {group_col} ===")
        run_one_grouping(group_col)


if __name__ == "__main__":
    main()
