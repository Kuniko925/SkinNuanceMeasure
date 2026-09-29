import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent

files = {
    ("BCN", 6): "results_eval/group_summary_BCN_test_seed_level_ita_skin_type.csv",
    ("BCN", 3): "results_eval/group_summary_BCN_test_seed_level_skin_tone_group_id.csv",
    ("HAM", 6): "results_eval/group_summary_HAM_test_seed_level_ita_skin_type.csv",
    ("HAM", 3): "results_eval/group_summary_HAM_test_seed_level_skin_tone_group_id.csv",
}

def main():

    records = []

    for (dataset, n_groups), path in files.items():
        df = pd.read_csv(path)

        for model_name, mdf in df.groupby("model_name"):

            for metric in ["accuracy", "f1", "TPR_malignant"]:

                # max-min gap within each seed
                seed_gaps = (
                    mdf.groupby("seed")[metric]
                    .agg(lambda x: x.max() - x.min())
                )

                records.append({
                    "dataset": dataset,
                    "model": model_name,
                    "n_groups": n_groups,
                    "metric": metric,
                    "gap_mean_pp": seed_gaps.mean() * 100,
                    "gap_std_pp": seed_gaps.std(ddof=1) * 100,
                })

    result = pd.DataFrame(records)

    wide = result.pivot(
        index=["dataset", "model"],
        columns=["metric", "n_groups"],
        values="gap_mean_pp"
    )

    for metric in ["accuracy", "f1", "TPR_malignant"]:
        wide[(metric, "increase")] = (
            wide[(metric, 6)] - wide[(metric, 3)]
        )

    print(wide.round(1))

    # 見やすい列順にする
    wide = wide[
        [
            ("accuracy", 3),
            ("accuracy", 6),
            ("accuracy", "increase"),
            ("f1", 3),
            ("f1", 6),
            ("f1", "increase"),
            ("TPR_malignant", 3),
            ("TPR_malignant", 6),
            ("TPR_malignant", "increase"),
        ]
    ]

    out = wide.reset_index()

    out.columns = [
        "dataset",
        "model",
        "accuracy_gap_3groups",
        "accuracy_gap_6groups",
        "accuracy_gap_increase",
        "f1_gap_3groups",
        "f1_gap_6groups",
        "f1_gap_increase",
        "TPR_malignant_gap_3groups",
        "TPR_malignant_gap_6groups",
        "TPR_malignant_gap_increase",
    ]

    out = out.round(1)

    output_path = (
            PROJECT_ROOT
            / "results_eval/category_gap_comparison_3groups_vs_6groups.csv"
    )

    out.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(out)
    print(f"saved: {output_path}")

if __name__ == "__main__":
    main()