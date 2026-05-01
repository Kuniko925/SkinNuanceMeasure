from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import BayesianRidge
import seaborn as sns
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import modelbuilder as mb
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent

seed = 42
random.seed(seed)
np.random.seed(seed)

class BayesianRegression:
    def __init__(self):
        self.full_data = None
        self.X_post = None
        self.y_pred = None
        self.brr_poly = None
        self.mu_value = None
    def fit(self, X, y, degree):
        self.full_data = pd.DataFrame({"wd": X, "tp proba": y})
        X = X.values.reshape((-1, 1))

        self.brr_poly = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            BayesianRidge(),
        ).fit(X, y)

    def predict(self, start, stop, num_data):
        X_post = np.linspace(start, stop, num_data)
        X_post = X_post.reshape((-1, 1))
        y_pred, y_std = self.brr_poly.predict(X_post, return_std=True)
        self.X_post = X_post
        self.y_pred = y_pred
        self.mu_value = np.mean(y_pred)
        return self.y_pred

    def display(self, ax, title, color="blue"):
        sns.scatterplot(
            data=self.full_data,
            x="wd",
            y="tp proba",
            color="grey",
            alpha=0.5,
            ax=ax,
            label="Observation",
        )
        ax.plot(self.X_post, self.y_pred, color=color, label="Estimation")
        ax.axhline(
            self.full_data["tp proba"].mean(),
            color="red",
            linestyle="--",
            label="Mean",
        )
        ax.set_xlabel("WD")
        ax.set_ylabel("TPR Probability")
        ax.set_title(title)
        ax.legend(loc="lower left")
        return ax

    def save_model(self, filepath):
        if self.brr_poly is not None:
            saved_obj = {
                "brr_poly": self.brr_poly,
                "mu_value": self.mu_value,
            }
            with open(filepath, "wb") as f:
                pickle.dump(saved_obj, f)
            print(f"Model saved {filepath}")

def make_binned_tpr(df_true, n_bins=10):
    df_true = df_true.copy()

    df_true["tp"] = (
        df_true["pred"].astype(str) == df_true["label"].astype(str)
    ).astype(float)

    df_true["wd_bin"] = pd.qcut(
        df_true["wd"],
        q=n_bins,
        duplicates="drop"
    )

    binned = (
        df_true
        .groupby("wd_bin", observed=True)
        .agg(
            wd=("wd", "mean"),
            tpr=("tp", "mean"),
            count=("tp", "size")
        )
        .reset_index(drop=True)
    )

    return binned

def main():
    db_names = ['milk10k/dermoscopic', 'milk10k/clinical', 'UTKFace']
    #db_names = ['UTKFace']
    model_names = list(mb.MODEL_REGISTRY.keys())

    for db_name in db_names:
        for model_name in model_names:
            filepath = PROJECT_ROOT / f"skin/{db_name}/results/baseline_{model_name}_valid_results.csv"
            df = pd.read_csv(filepath)

            df = df.copy()
            df["label"] = df["label"].astype(str)
            df["wd"] = df["wd"].astype(float)

            all_classes = sorted(df["label"].unique())

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()

            for ax, cls in zip(axes, all_classes):
                df_true = df[df["label"] == cls].copy()
                binned = make_binned_tpr(df_true, n_bins=25)
                X = binned["wd"]
                y = binned["tpr"]

                degree = 3
                br = BayesianRegression()
                br.fit(X, y, degree)
                br.predict(start=X.min(), stop=X.max(), num_data=1000)
                br.display(ax, title=f"Class {cls} Equal Opportunity")
                br_filepath = PROJECT_ROOT / f"skin/{db_name}/bayes/{model_name}_{cls}.pkl"
                br.save_model(br_filepath)

            for ax in axes[len(all_classes):]:
                ax.axis("off")

            plt.tight_layout()
            outpath = PROJECT_ROOT / f"skin/{db_name}/results/{model_name}_bayesian_poly_by_class.png"
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close()


if __name__ == '__main__':
    main()