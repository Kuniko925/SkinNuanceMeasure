from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import TRAIN_MODES, DATASETS, SEEDS, MODEL_REGISTRY

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent

def _rates_binary_ovr(y_true, y_pred, positive_class):
    yt = (y_true == positive_class)
    yp = (y_pred == positive_class)

    TP = np.sum(yt & yp)
    FN = np.sum(yt & (~yp))
    FP = np.sum((~yt) & yp)
    TN = np.sum((~yt) & (~yp))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    return TPR, FPR


def fairness_summary_multiclass_ovr_macro(df, label_col="label", pred_col="pred", group_col="ita_skin_type", f1_average="macro",):
    y_true = df[label_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    g = df[group_col].to_numpy()

    overall_omr = float(np.mean(y_true != y_pred))

    classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    groups = np.unique(g)
    grp_masks = {grp: (g == grp) for grp in groups}

    eopp_per_class = []
    eodds_per_class = []

    for c in classes:
        tpr, fpr = {}, {}

        for grp in groups:
            m = grp_masks[grp]
            TPR, FPR = _rates_binary_ovr(
                y_true[m],
                y_pred[m],
                positive_class=c
            )

            tpr[grp], fpr[grp] = TPR, FPR

        tpr_vals = np.array([tpr[grp] for grp in groups], dtype=float)

        eopp_c = np.nanmax(tpr_vals) - np.nanmin(tpr_vals)

        if not np.isnan(eopp_c):
            eopp_per_class.append(float(eopp_c))

        max_gap_val = 0.0
        any_valid = False

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                gi, gj = groups[i], groups[j]

                if (
                    np.isnan(tpr[gi]) or
                    np.isnan(tpr[gj]) or
                    np.isnan(fpr[gi]) or
                    np.isnan(fpr[gj])
                ):
                    continue

                gap = max(
                    abs(tpr[gi] - tpr[gj]),
                    abs(fpr[gi] - fpr[gj])
                )

                max_gap_val = max(max_gap_val, gap)
                any_valid = True

        if any_valid:
            eodds_per_class.append(float(max_gap_val))

    eopp_macro = (
        float(np.mean(eopp_per_class))
        if eopp_per_class else None
    )

    eodds_macro = (
        float(np.mean(eodds_per_class))
        if eodds_per_class else None
    )

    group_rows = []

    for grp in groups:
        m = grp_masks[grp]

        yt = y_true[m]
        yp = y_pred[m]

        acc = accuracy_score(yt, yp)

        grp_omr = 1.0 - acc

        tprs, fprs = [], []

        for c in classes:
            TPR, FPR = _rates_binary_ovr(
                yt,
                yp,
                positive_class=c
            )

            if not np.isnan(TPR):
                tprs.append(TPR)

            if not np.isnan(FPR):
                fprs.append(FPR)

        group_rows.append({
            "group": grp,
            "n_samples": int(m.sum()),
            "accuracy": acc,
            "group_misclassification_rate": grp_omr,
            "f1": f1_score(
                yt,
                yp,
                average=f1_average,
                zero_division=0
            ),
            "macro_TPR": float(np.mean(tprs)) if tprs else None,
            "macro_FPR": float(np.mean(fprs)) if fprs else None,
        })

    group_df = (
        pd.DataFrame(group_rows)
        .sort_values("group")
        .reset_index(drop=True)
    )

    if len(group_df):
        gap = float(
            group_df["group_misclassification_rate"].max()
            - group_df["group_misclassification_rate"].min()
        )
    else:
        gap = None

    return {
        "overall_misclassification_rate": overall_omr,
        "equal_opportunity_diff_macro_ovr": eopp_macro,
        "equalized_odds_diff_macro_ovr": eodds_macro,
        "group_misclassification_rate_gap": gap,
        "group_metrics": group_df,
    }


def main():

    train_modes = TRAIN_MODES
    db_names = DATASETS
    model_names = list(MODEL_REGISTRY.keys())
    eval_types = ["test"]
    group_col = "ita_skin_type"
    #group_col = "skin_tone_group_id"

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

                        df = (
                            pd.read_csv(filepath)
                            .dropna(subset=[group_col])
                        )

                        res = fairness_summary_multiclass_ovr_macro(
                            df,
                            group_col=group_col
                        )

                        seed_metrics.append({
                            "seed": seed,
                            "OMR": res["overall_misclassification_rate"],
                            "Eopp": res["equal_opportunity_diff_macro_ovr"],
                            "Eodds": res["equalized_odds_diff_macro_ovr"],
                        })

                        tmp_df = res["group_metrics"].copy()

                        tmp_df["dataset"] = db_name
                        tmp_df["model_name"] = model_name
                        tmp_df["train_mode"] = train_mode
                        tmp_df["seed"] = seed

                        group_seed_records.append(tmp_df)

                    # =========================
                    # 3 seed of mean / std
                    # =========================
                    seed_df = pd.DataFrame(seed_metrics)

                    if len(seed_df) == 0:
                        continue

                    label = train_mode

                    for metric in ["OMR", "Eopp", "Eodds"]:

                        row[(label, f"{metric}_mean")] = (
                            seed_df[metric].mean()
                        )

                        row[(label, f"{metric}_std")] = (
                            seed_df[metric].std(ddof=1)
                        )

                records.append(row)

            # ==================================================
            # fairness summary
            # ==================================================
            summary_df = pd.DataFrame(records)
            summary_df.columns = pd.MultiIndex.from_tuples(
                summary_df.columns
            )

            output_path = (
                PROJECT_ROOT
                / f"results_eval/fairness_summary_{db_name}_{eval_type}_wide_{group_col}.csv"
            )

            summary_df.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig"
            )

            # ==================================================
            # group metrics
            # seed-level
            # ==================================================
            if group_seed_records:

                group_seed_df = pd.concat(
                    group_seed_records,
                    ignore_index=True
                )

                # seed
                seed_group_path = (
                    PROJECT_ROOT
                    / f"results_eval/group_summary_{db_name}_{eval_type}_seed_level_{group_col}.csv"
                )

                group_seed_df.to_csv(
                    seed_group_path,
                    index=False,
                    encoding="utf-8-sig"
                )

                # ----------------------------------------------
                # group × model × train_mode
                # seed avg and std
                # ----------------------------------------------
                metrics = [
                    "accuracy",
                    "group_misclassification_rate",
                    "f1",
                    "macro_TPR",
                    "macro_FPR",
                ]

                agg_dict = {}

                for metric in metrics:
                    agg_dict[f"{metric}_mean"] = (
                        metric,
                        "mean"
                    )
                    agg_dict[f"{metric}_std"] = (
                        metric,
                        "std"
                    )

                group_summary_df = (
                    group_seed_df
                    .groupby(
                        [
                            "dataset",
                            "model_name",
                            "train_mode",
                            "group",
                        ],
                        as_index=False
                    )
                    .agg(
                        n_seeds=("seed", "nunique"),
                        **agg_dict
                    )
                )

                performance_csv_path = (
                    PROJECT_ROOT
                    / f"results_eval/group_summary_{db_name}_{eval_type}_all_modes_{group_col}.csv"
                )

                group_summary_df.to_csv(
                    performance_csv_path,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(f"saved: {seed_group_path}")
                print(f"saved: {performance_csv_path}")

            print(f"saved: {output_path}")



if __name__ == "__main__":
    main()