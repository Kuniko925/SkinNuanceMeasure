from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.special import expit, logit
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

DATASETS = [
    "HAM",
    "BCN",
]

MODELS = [
    "ResNet50",
    "MobileNetV2",
    "SwinT",
]

SEEDS = [
    12,
    123,
    1234,
]

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant",
}

DEGREE = 3
N_BINS = 25


def safe_logit(y, eps=1e-6):
    """
    Convert probabilities to logit space safely.
    """
    y = np.asarray(y, dtype=float)
    y = np.clip(y, eps, 1.0 - eps)

    return logit(y)


class BayesianRegression:

    def __init__(self):
        self.model = None
        self.mu_value = None

    def fit(self, X, y, degree=3):

        X = np.asarray(X, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float)

        self.model = TransformedTargetRegressor(
            regressor=make_pipeline(
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                ),
                StandardScaler(),
                BayesianRidge(),
            ),
            func=safe_logit,
            inverse_func=expit,
            check_inverse=False,
        )

        self.model.fit(X, y)

    def predict(self, X):

        X = np.asarray(X, dtype=float).reshape(-1, 1)

        return self.model.predict(X)

    def estimate_mean(self, start, stop, num_points=1000):

        X_grid = np.linspace(
            start,
            stop,
            num_points
        )

        y_pred = self.predict(X_grid)

        self.mu_value = float(
            np.mean(y_pred)
        )

        return (
            X_grid,
            y_pred
        )

    def save(self, filepath):

        saved_obj = {
            "brr_poly": self.model,
            "mu_value": self.mu_value,
        }

        with open(filepath, "wb") as f:
            pickle.dump(
                saved_obj,
                f
            )

        print(
            f"Saved: {filepath}"
        )


def make_binned_performance(
    df,
    n_bins=25
):
    """
    Calculate class-conditional classification performance
    for samples with similar Wasserstein distances.
    """

    df = df.copy()

    # Correct classification: 1
    # Incorrect classification: 0
    df["correct"] = (
        df["pred"].astype(int)
        == df["binary_label_id"].astype(int)
    ).astype(float)

    # qcut requires enough unique WD values
    n_unique = df["wd"].nunique()

    actual_bins = min(
        n_bins,
        n_unique,
        len(df)
    )

    if actual_bins < 2:
        raise ValueError(
            "Not enough unique WD values "
            "to construct bins."
        )

    df["wd_bin"] = pd.qcut(
        df["wd"],
        q=actual_bins,
        duplicates="drop",
    )

    binned = (
        df
        .groupby(
            "wd_bin",
            observed=True
        )
        .agg(
            wd=("wd", "mean"),
            performance=("correct", "mean"),
            count=("correct", "size"),
        )
        .reset_index(drop=True)
    )

    return binned


def validate_columns(df, filepath):

    required_columns = {
        "binary_label_id",
        "pred",
        "wd",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{filepath} is missing columns: "
            f"{sorted(missing)}"
        )


def fit_regressor(
    df,
    cls,
    degree=3,
    n_bins=25
):

    df_class = df[
        df["binary_label_id"] == cls
    ].copy()

    if len(df_class) == 0:
        raise ValueError(
            f"No samples found for class {cls}"
        )

    binned = make_binned_performance(
        df_class,
        n_bins=n_bins
    )

    X = binned["wd"].to_numpy()
    y = binned["performance"].to_numpy()

    br = BayesianRegression()

    br.fit(
        X,
        y,
        degree=degree
    )

    X_grid, y_grid = br.estimate_mean(
        X.min(),
        X.max()
    )

    return (
        br,
        binned,
        X_grid,
        y_grid
    )


def plot_regression(
    ax,
    binned,
    X_grid,
    y_grid,
    cls
):

    ax.scatter(
        binned["wd"],
        binned["performance"],
        alpha=0.7,
        label="Observed",
    )

    ax.plot(
        X_grid,
        y_grid,
        label="Bayesian regression",
    )

    ax.axhline(
        binned["performance"].mean(),
        linestyle="--",
        label="Mean",
    )

    class_name = CLASS_NAMES.get(
        cls,
        str(cls)
    )

    ax.set_title(
        f"{class_name} (class {cls})"
    )

    ax.set_xlabel(
        "Wasserstein distance"
    )

    ax.set_ylabel(
        "Classification performance"
    )

    ax.set_ylim(
        -0.05,
        1.05
    )

    ax.legend()


def process_run(
    db_name,
    model_name,
    seed
):

    result_path = (
        RESULTS_DIR
        / (
            f"{db_name}_baseline_"
            f"{model_name}_seed{seed}_valid.csv"
        )
    )

    if not result_path.exists():
        print(
            f"Skip: {result_path} "
            f"does not exist."
        )

        return

    print(
        "\n"
        f"Dataset: {db_name} | "
        f"Model: {model_name} | "
        f"Seed: {seed}"
    )

    df = pd.read_csv(
        result_path
    )

    validate_columns(
        df,
        result_path
    )

    df["binary_label_id"] = (
        df["binary_label_id"]
        .astype(int)
    )

    df["pred"] = (
        df["pred"]
        .astype(int)
    )

    df["wd"] = (
        df["wd"]
        .astype(float)
    )

    bayes_dir = (
        PROJECT_ROOT
        / "skin"
        / db_name
        / "bayes"
    )

    bayes_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    classes = sorted(
        df["binary_label_id"].unique()
    )

    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=(10, 4)
    )

    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(
        axes,
        classes
    ):

        (
            br,
            binned,
            X_grid,
            y_grid,
        ) = fit_regressor(
            df,
            cls=cls,
            degree=DEGREE,
            n_bins=N_BINS,
        )

        # --------------------------------------------------
        # Save Bayesian regressor
        # --------------------------------------------------

        model_path = (
            bayes_dir
            / (
                f"{model_name}_"
                f"seed{seed}_"
                f"{cls}.pkl"
            )
        )

        br.save(
            model_path
        )

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        plot_regression(
            ax,
            binned,
            X_grid,
            y_grid,
            cls,
        )

    fig.suptitle(
        (
            f"{db_name} | "
            f"{model_name} | "
            f"seed {seed}"
        )
    )

    fig.tight_layout()

    figure_path = (
        bayes_dir
        / (
            f"{model_name}_"
            f"seed{seed}_"
            f"bayesian_regression.png"
        )
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Figure saved: {figure_path}"
    )


def main():

    for seed in SEEDS:

        for db_name in DATASETS:

            for model_name in MODELS:

                process_run(
                    db_name=db_name,
                    model_name=model_name,
                    seed=seed,
                )


if __name__ == "__main__":
    main()