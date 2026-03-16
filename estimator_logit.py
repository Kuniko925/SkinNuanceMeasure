import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import dcor

PROJECT_ROOT = Path(__file__).parent # Path to folder opened file

def compute_test_results(df):
    df = df.copy()
    df["label"] = df["label"].astype(str)
    df["pred"] = df["pred"].astype(str)

    all_classes = sorted(df["label"].unique())
    rows = []
    cors = []

    for cls in all_classes:

        df_true = df[df["label"] == cls].copy()
        df_true["tp"] = (df_true["pred"] == cls).astype(float)

        X = sm.add_constant(df_true["wd"]).astype(float)
        y = df_true["tp"].astype(float)

        cor = dcor.distance_correlation(X, y)
        cors.append(cor)

        try:
            model = sm.Logit(y, X).fit(disp=False)
            ci = model.conf_int()

            rows.append({
                "class": cls,
                "n": int(model.nobs),
                "tpr_mean": df_true["tp"].mean(),
                "coef_const": model.params["const"],
                "coef_wd": model.params["wd"],
                "se_wd": model.bse["wd"],
                "z_wd": model.tvalues["wd"],
                "p_wd": model.pvalues["wd"],
                "ci_low_wd": ci.loc["wd", 0],
                "ci_high_wd": ci.loc["wd", 1],
                "pseudo_r2": model.prsquared,
                "llr_pvalue": model.llr_pvalue,
                "note": ""
            })

        except Exception as e:
            rows.append({
                "class": cls,
                "n": len(df_true),
                "tpr_mean": df_true["tp"].mean(),
                "coef_const": np.nan,
                "coef_wd": np.nan,
                "se_wd": np.nan,
                "z_wd": np.nan,
                "p_wd": np.nan,
                "ci_low_wd": np.nan,
                "ci_high_wd": np.nan,
                "pseudo_r2": np.nan,
                "llr_pvalue": np.nan,
                "note": str(e)
            })

    results_df = pd.DataFrame(rows)
    cors_df = pd.DataFrame(cors)
    return results_df, cors_df


def plot_tpr_lowess_by_class(df, outpath):
    df = df.copy()
    df["label"] = df["label"].astype(str)
    df["pred"] = df["pred"].astype(str)

    all_classes = sorted(df["label"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, cls in zip(axes, all_classes):
        df_true = df[df["label"] == cls].copy()
        df_true["tp"] = (df_true["pred"] == cls).astype(int)

        sns.regplot(
            x="wd",
            y="tp",
            data=df_true,
            lowess=True,
            scatter_kws={"alpha": 0.35, "s": 18},
            line_kws={"linewidth": 2},
            ax=ax
        )

        n = len(df_true)
        tpr_mean = df_true["tp"].mean()

        ax.set_title(f"Class {cls} (n={n}, mean TPR={tpr_mean:.3f})")
        ax.set_xlabel("Wasserstein Distance")
        ax.set_ylabel("TP / TPR proxy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)

    for ax in axes[len(all_classes):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def main():

    db_name = 'milk10k/dermoscopic'
    model_names = ['MobileNetV3', 'MobileVitV2', 'SwinTV2']

    for model_name in model_names:

        filepath = PROJECT_ROOT / f"skin/{db_name}/results/{model_name}_valid_results.csv"
        usecols = ["filepath", "label", "pred", "skin tone", "wd"]
        df = pd.read_csv(filepath, usecols=usecols)
        res, cor = compute_test_results(df)

        respath = PROJECT_ROOT / f"skin/{db_name}/results/{model_name}_estimator.csv"
        res.to_csv(respath, index=False)

        respath = PROJECT_ROOT / f"skin/{db_name}/results/{model_name}_cor.csv"
        cor.to_csv(respath, index=False)

        outpath = PROJECT_ROOT / f"skin/{db_name}/results/{model_name}_wd_tpr_lowess_by_class.png"
        plot_tpr_lowess_by_class(df, outpath)

if __name__== "__main__":

    main()