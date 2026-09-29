from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.special import logit
from bayesian_spline import BayesianBinomialSpline

from config import DATASETS, MODEL_REGISTRY, SEEDS, TRAIN_MODES

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant",
}

N_BINS = 25
N_KNOTS = 7
SMOOTHNESS = 10.0
COEFFICIENT_SCALE = 5.0


def safe_logit(y, eps=1e-6):
    """
    Retained for loading older polynomial-regression pickle files only.
    """
    y = np.asarray(y, dtype=float)
    y = np.clip(y, eps, 1.0 - eps)

    return logit(y)


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
            correct=("correct", "sum"),
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
    correct = binned["correct"].to_numpy()
    count = binned["count"].to_numpy()

    br = BayesianBinomialSpline(
        n_knots=N_KNOTS, smoothness=SMOOTHNESS,
        coefficient_scale=COEFFICIENT_SCALE,
    )

    br.fit(
        X,
        correct,
        count
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
    cls,
    lower,
    upper,
):

    ax.scatter(
        binned["wd"],
        binned["performance"],
        alpha=0.7,
        s=20 + 60 * binned["count"] / binned["count"].max(),
        label="Observed (size = count)",
    )

    ax.plot(
        X_grid,
        y_grid,
        label="Binomial spline: posterior mean",
    )

    ax.fill_between(
        X_grid, lower, upper, alpha=0.2,
        label="95% pointwise CrI (Laplace)",
    )

    ax.axhline(
        binned["correct"].sum() / binned["count"].sum(),
        linestyle="--",
        label="Pooled observed accuracy",
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


def load_validation_results(db_name, train_mode, model_name):
    """Pool every configured seed's validation predictions before binning.

    Each prediction row has equal weight, including repeated images across
    seeds. Test predictions and seeds outside config.SEEDS are not included.
    """
    if not SEEDS:
        raise ValueError("config.SEEDS must contain at least one seed.")

    paths = [
        RESULTS_DIR / f"{db_name}_{train_mode}_{model_name}_seed{seed}_valid.csv"
        for seed in SEEDS
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing validation results for configured seeds:\n" + "\n".join(missing)
        )

    frames = []
    for seed, path in zip(SEEDS, paths):
        df = pd.read_csv(path)
        validate_columns(df, path)
        df = df[["binary_label_id", "pred", "wd"]].copy()
        if df.empty:
            raise ValueError(f"{path} contains no validation predictions.")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError(f"{path} contains missing or non-finite label/pred/WD values.")
        for col in ["binary_label_id", "pred"]:
            if not df[col].isin(CLASS_NAMES).all():
                raise ValueError(f"{path}: {col} must contain only 0 or 1.")
            df[col] = df[col].astype(int)
        df["seed"] = seed
        frames.append(df)
        print(f"Loaded: {path} ({len(df)} rows)")

    return pd.concat(frames, ignore_index=True)


def process_run(db_name, train_mode, model_name):
    df = load_validation_results(db_name, train_mode, model_name)
    print(
        f"\nDataset: {db_name} | Train mode: {train_mode} | "
        f"Model: {model_name} | Pooled seeds: {SEEDS} | Rows: {len(df)}"
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
            n_bins=N_BINS,
        )

        # --------------------------------------------------
        # Save Bayesian regressor
        # --------------------------------------------------

        model_path = (
            bayes_dir
            / (
                f"{train_mode}_{model_name}_"
                f"pooled_seeds_binomial_spline_"
                f"{cls}.pkl"
            )
        )

        br.save(model_path, metadata={
            "dataset": db_name, "train_mode": train_mode,
            "model_name": model_name, "seeds": list(SEEDS),
            "class": int(cls), "split": "valid",
            "n_knots": N_KNOTS, "smoothness": SMOOTHNESS,
            "coefficient_scale": COEFFICIENT_SCALE,
        })
        lower, upper = br.predict_interval(X_grid)
        prefix = f"{train_mode}_{model_name}_pooled_seeds_binomial_spline_{cls}"
        binned.to_csv(bayes_dir / f"{prefix}_bins.csv", index=False)
        pd.DataFrame({
            "wd": X_grid, "posterior_mean": y_grid,
            "lower_95": lower, "upper_95": upper,
        }).to_csv(bayes_dir / f"{prefix}_curve.csv", index=False)

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        plot_regression(
            ax,
            binned,
            X_grid,
            y_grid,
            cls,
            lower,
            upper,
        )

    fig.suptitle(
        (
            f"{db_name} | "
            f"{train_mode} | "
            f"{model_name} | "
            f"pooled seeds {SEEDS}"
        )
    )

    fig.text(
        0.5, 0.01,
        "Intervals assume independent binomial trials; repeated-image/seed dependence is not modeled.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    figure_path = (
        bayes_dir
        / (
            f"{train_mode}_{model_name}_"
            f"pooled_seeds_binomial_spline_"
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

    for db_name in DATASETS:
        for train_mode in TRAIN_MODES:
            for model_name in MODEL_REGISTRY:

                process_run(
                    db_name=db_name,
                    train_mode=train_mode,
                    model_name=model_name,
                )


if __name__ == "__main__":
    # Use canonical module names for reusable saved models.
    from bayesian import main as run_main

    run_main()
