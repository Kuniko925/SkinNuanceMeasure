from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import BayesianRidge
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent

class BayesianRegression:
    def __init__(self):
        self.full_data = None
        self.X_post = None
        self.y_pred = None
        self.y_std = None
        self.brr_poly = None

    def fit(self, X, y, degree):
        self.full_data = pd.DataFrame({"wd": X, "tp": y})
        X = X.values.reshape((-1, 1))

        self.brr_poly = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            BayesianRidge(),
        ).fit(X, y)

    def predict(self, start, stop, num_data):
        X_post = np.linspace(start, stop, num_data)
        X_post = X_post.reshape((-1, 1))
        y_pred, y_std = self.brr_poly.predict(X_post, return_std=True)
        self.X_post = X_post
        self.y_pred = y_pred
        self.y_std = y_std
        return self.y_pred, self.y_std

    def display(self, ax, title, color="blue", std_color="lightblue"):
        sns.scatterplot(
            data=self.full_data,
            x="wd",
            y="tp",
            color="grey",
            alpha=0.5,
            ax=ax,
            label="Observation",
        )
        ax.plot(self.X_post, self.y_pred, color=color, label="Estimation")
        ax.fill_between(
            self.X_post.ravel(),
            np.clip(self.y_pred - self.y_std, 0, 1),
            np.clip(self.y_pred + self.y_std, 0, 1),
            color=std_color,
            alpha=0.3,
        )
        ax.axhline(
            self.full_data["tp"].mean(),
            color="red",
            linestyle="--",
            label="Mean",
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("WD")
        ax.set_ylabel("TPR proxy")
        ax.set_title(title)
        ax.legend(loc="lower left")
        return ax

    def save_model(self, filepath):
        if self.brr_poly is not None:
            with open(filepath, "wb") as f:
                pickle.dump(self.brr_poly, f)
                print(f"Model saved")

def main():
    db_name = 'milk10k/dermoscopic'
    model_names = ['MobileNetV3', 'MobileVitV2', 'SwinTV2']
    model_name = model_names[0]

    filepath = PROJECT_ROOT / f"skin/{db_name}/results/{model_name}_valid_results.csv"
    usecols = ["filepath", "label", "pred", "skin tone", "wd"]
    df = pd.read_csv(filepath, usecols=usecols)

    df = df.copy()
    df["label"] = df["label"].astype(str)
    df["pred"] = df["pred"].astype(str)
    df["wd"] = df["wd"].astype(float)

    all_classes = sorted(df["label"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, cls in zip(axes, all_classes):
        df_true = df[df["label"] == cls].copy()
        df_true["tp"] = (df_true["pred"] == cls).astype(float)

        X = df_true["wd"]
        y = df_true["tp"]

        degree = 2
        br = BayesianRegression()
        br.fit(X, y, degree)
        br.predict(start=X.min(), stop=X.max(), num_data=200)
        br.display(ax, title=f"Class {cls}")

    for ax in axes[len(all_classes):]:
        ax.axis("off")

    plt.tight_layout()
    outpath = PROJECT_ROOT / f"skin/{db_name}/results/{model_name}_bayesian_poly_by_class.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    main()