import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import modelbuilder as mb
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent


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

def fairness_summary_multiclass_ovr_macro(
    df,
    label_col="label",
    pred_col="pred",
    group_col="skin tone",
    f1_average="macro",
):
    y_true = df[label_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    g      = df[group_col].to_numpy()

    # overall misclassification rate
    overall_omr = float(np.mean(y_true != y_pred))

    classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    groups  = np.unique(g)
    grp_masks = {grp: (g == grp) for grp in groups}

    # Equal Opportunity / Equalized Odds (macro over classes, one-vs-rest)
    eopp_per_class = []
    eodds_per_class = []

    for c in classes:
        tpr, fpr = {}, {}
        for grp in groups:
            m = grp_masks[grp]
            TPR, FPR = _rates_binary_ovr(y_true[m], y_pred[m], positive_class=c)
            tpr[grp], fpr[grp] = TPR, FPR

        # EOpp: max(TPR)-min(TPR)
        tpr_vals = np.array([tpr[grp] for grp in groups], dtype=float)
        eopp_c = np.nanmax(tpr_vals) - np.nanmin(tpr_vals)
        if not np.isnan(eopp_c):
            eopp_per_class.append(float(eopp_c))

        # EOdds: max pairwise gap over {TPR, FPR}
        max_gap_val = 0.0
        any_valid = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                gi, gj = groups[i], groups[j]
                if np.isnan(tpr[gi]) or np.isnan(tpr[gj]) or np.isnan(fpr[gi]) or np.isnan(fpr[gj]):
                    continue
                gap = max(abs(tpr[gi] - tpr[gj]), abs(fpr[gi] - fpr[gj]))
                max_gap_val = max(max_gap_val, gap)
                any_valid = True
        if any_valid:
            eodds_per_class.append(float(max_gap_val))

    eopp_macro = float(np.mean(eopp_per_class)) if eopp_per_class else None
    eodds_macro = float(np.mean(eodds_per_class)) if eodds_per_class else None

    # group-level summary
    group_rows = []
    for grp in groups:
        m = grp_masks[grp]
        yt, yp = y_true[m], y_pred[m]

        acc = accuracy_score(yt, yp)
        grp_omr = 1.0 - acc  # group-wise misclassification rate

        # macro TPR/FPR inside group (ovr over classes)
        tprs, fprs = [], []
        for c in classes:
            TPR, FPR = _rates_binary_ovr(yt, yp, positive_class=c)
            if not np.isnan(TPR): tprs.append(TPR)
            if not np.isnan(FPR): fprs.append(FPR)

        group_rows.append({
            "group": grp,
            "n_samples": int(m.sum()),
            "accuracy": acc,
            "group_misclassification_rate": grp_omr,
            "f1": f1_score(yt, yp, average=f1_average, zero_division=0),
            "macro_TPR": float(np.mean(tprs)) if tprs else None,
            "macro_FPR": float(np.mean(fprs)) if fprs else None,
        })

    group_df = pd.DataFrame(group_rows).sort_values("group").reset_index(drop=True)

    # disparity of misclassification rate across groups (max-min)
    if len(group_df):
        gap = float(group_df["group_misclassification_rate"].max() - group_df["group_misclassification_rate"].min())
    else:
        gap = None

    return {
        "overall_misclassification_rate": overall_omr,  # P(pred!=label)
        "equal_opportunity_diff_macro_ovr": eopp_macro,
        "equalized_odds_diff_macro_ovr": eodds_macro,
        "group_misclassification_rate_gap": gap,        # max_g - min_g of P(pred!=label | A=g)
        "group_metrics": group_df,
    }

def main():

    #train_mode = 'baseline'
    #train_mode = 'regularization'
    #train_mode = 'reweight'
    train_mode = 'individual'

    db_names = ['UTKFace', 'milk10k/dermoscopic', 'milk10k/clinical']
    model_names = list(mb.MODEL_REGISTRY.keys())
    eval_types = ['test']

    for db_name in db_names:
        for eval_type in eval_types:
            summary_csv_path = PROJECT_ROOT / f"skin/{db_name}/results/fairness_summary_{eval_type}_{train_mode}.csv"
            performance_csv_path = PROJECT_ROOT / f"skin/{db_name}/results/group_summary_{eval_type}_{train_mode}.csv"
            records = []
            performances_df = pd.DataFrame()

            for model_name in model_names:
                filepath = PROJECT_ROOT / f"skin/{db_name}/results/{train_mode}_{model_name}_{eval_type}_results.csv"
                df = pd.read_csv(filepath).dropna(subset=["skin tone"])
                res = fairness_summary_multiclass_ovr_macro(df, group_col="skin tone")

                lines = [
                    f"{model_name}",
                    f"OMR: {res['overall_misclassification_rate']}",
                    f"group OMR gap: {res['group_misclassification_rate_gap']}",
                    f"EOpp (macro ovr): {res['equal_opportunity_diff_macro_ovr']}",
                    f"EOdds (macro ovr): {res['equalized_odds_diff_macro_ovr']}",
                    f"{res['group_metrics']}",
                    "\n" + "=" * 50 + "\n"
                ]

                text = "\n".join(lines)
                print(text)

                records.append({
                    "model_name": model_name,
                    "OMR": res["overall_misclassification_rate"],
                    "group_OMR_gap": res["group_misclassification_rate_gap"],
                    "EOpp_macro_ovr": res["equal_opportunity_diff_macro_ovr"],
                    "EOdds_macro_ovr": res["equalized_odds_diff_macro_ovr"],
                })

                tmp_df = res['group_metrics']
                tmp_df['model_name'] = model_name
                performances_df = pd.concat([performances_df, tmp_df], ignore_index=True)


        summary_df = pd.DataFrame(records)
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
        print(f"saved: {summary_csv_path}")
        performances_df.to_csv(performance_csv_path, index=False, encoding="utf-8")
        print(f"saved: {performance_csv_path}")


if __name__ == "__main__":
    main()