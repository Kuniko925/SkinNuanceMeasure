from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from config import DATASETS, IMAGE_EXTENSIONS

from skincolor import IndividualTypologyAngle

PROJECT_ROOT = Path(__file__).resolve().parent

N_REFERENCE_PIXELS_PER_IMAGE = 1000
REFERENCE_SEED = 42

def get_image_id(row):
    """
    Get image ID from the dataframe.
    """

    for col in [
        "isic_id",
        "image_id",
        "ISIC_id",
    ]:
        if col in row.index:
            return str(row[col])

    raise KeyError(
        "No image ID column found. "
        "Expected one of: isic_id, image_id, ISIC_id"
    )


def select_reference_images(
    df_train,
    dataset_root,
):
    """
    Use all Type 6 images in the training set
    as reference images.
    """

    if "ita_skin_type" not in df_train.columns:
        raise KeyError(
            "'ita_skin_type' not found in training dataframe."
        )

    reference_df = df_train[
        df_train["ita_skin_type"] == 6
    ].copy()

    reference_df = reference_df.reset_index(
        drop=True
    )

    if len(reference_df) == 0:
        raise ValueError(
            "No Type 6 images found in training set."
        )

    rows = []

    for _, row in reference_df.iterrows():

        skin_path, fov_path = (
            get_skin_and_fov_paths(
                dataset_root,
                row,
            )
        )

        rows.append(
            {
                "isic_id": get_image_id(row),
                "skin_filepath": str(
                    skin_path
                ),
                "fov_filepath": str(
                    fov_path
                ),
                "ita": float(
                    row["ita"]
                ),
                "ita_valid_fraction": float(
                    row["ita_valid_fraction"]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_reference_distribution(
    reference_df,
):
    """
    Build reference distribution from all
    training-set Type 6 images.

    The same number of pixels is sampled
    from every image so that images with
    larger visible skin regions do not
    dominate the reference distribution.
    """

    rng = np.random.default_rng(
        REFERENCE_SEED
    )

    distributions = []

    for _, row in reference_df.iterrows():

        ita_values = (
            compute_ita_distribution(
                Path(
                    row["skin_filepath"]
                ),
                Path(
                    row["fov_filepath"]
                ),
            )
        )

        if (
            len(ita_values)
            < N_REFERENCE_PIXELS_PER_IMAGE
        ):
            raise ValueError(
                f"Too few valid skin pixels: "
                f"{row['isic_id']} "
                f"({len(ita_values)} pixels)"
            )

        sampled = rng.choice(
            ita_values,
            size=N_REFERENCE_PIXELS_PER_IMAGE,
            replace=False,
        )

        distributions.append(
            sampled
        )

    reference_distribution = (
        np.concatenate(
            distributions
        )
    )

    return reference_distribution

def find_image(
    dataset_root,
    image_id,
    subdir,
):
    image_dir = (
        dataset_root
        / subdir
    )

    for ext in IMAGE_EXTENSIONS:
        filepath = (
            image_dir
            / f"{image_id}{ext}"
        )

        if filepath.exists():
            return filepath

    raise FileNotFoundError(
        f"{subdir} image not found: "
        f"{image_id}"
    )

def get_skin_and_fov_paths(
    dataset_root,
    row,
):
    image_id = get_image_id(row)

    skin_path = find_image(
        dataset_root,
        image_id,
        "skin",
    )

    fov_path = find_image(
        dataset_root,
        image_id,
        "fov",
    )

    return skin_path, fov_path

def compute_ita_distribution(
    skin_filepath,
    fov_filepath,
):
    ita = IndividualTypologyAngle(
        str(skin_filepath),
        str(fov_filepath),
    )

    values = np.asarray(
        ita.compute_nuance_ita(),
        dtype=float,
    ).reshape(-1)

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise ValueError(
            f"No valid ITA pixels: "
            f"{skin_filepath}"
        )

    return values


def build_reference_distribution(
    reference_df
):

    distributions = []

    for _, row in reference_df.iterrows():

        ita_values = (
            compute_ita_distribution(
                Path(
                    row["skin_filepath"]
                ),
                Path(
                    row["fov_filepath"]
                ),
            )
        )

        distributions.append(
            ita_values
        )

    return np.concatenate(
        distributions
    )


def add_wasserstein_distance(
    df,
    dataset_root,
    reference_distribution,
):

    df = df.copy()

    distances = []
    medians = []
    skin_filepaths = []

    total = len(df)

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        skin_path, fov_path = (
            get_skin_and_fov_paths(
                dataset_root,
                row,
            )
        )

        ita_values = (
            compute_ita_distribution(
                skin_path,
                fov_path,
            )
        )

        distance = (
            wasserstein_distance(
                reference_distribution,
                ita_values,
            )
        )

        distances.append(
            float(distance)
        )

        medians.append(
            float(
                np.median(
                    ita_values
                )
            )
        )

        skin_filepaths.append(
            str(skin_path)
        )

        if (
            i % 500 == 0
            or i == total
        ):
            print(
                f"  {i}/{total}"
            )

    df["skin_filepath"] = (
        skin_filepaths
    )

    df["skin_ita_median"] = (
        medians
    )

    df["wd"] = (
        distances
    )

    return df


def process_dataset(
    dataset_name
):

    print(
        "\n"
        "====================================\n"
        f"{dataset_name}\n"
        "===================================="
    )

    dataset_root = (
        PROJECT_ROOT
        / "skin"
        / dataset_name
    )

    train_path = (
        dataset_root
        / "df_train.csv"
    )

    valid_path = (
        dataset_root
        / "df_valid.csv"
    )

    test_path = (
        dataset_root
        / "df_test.csv"
    )

    df_train = pd.read_csv(
        train_path
    )

    df_valid = pd.read_csv(
        valid_path
    )

    df_test = pd.read_csv(
        test_path
    )

    # --------------------------------------------------
    # 1. Define reference using training Type 6 only
    # --------------------------------------------------

    reference_df = (
        select_reference_images(
            df_train,
            dataset_root,
        )
    )

    print(
        "\nReference:"
    )

    print(
        "Training Type 6 images:",
        len(reference_df),
    )

    # --------------------------------------------------
    # 2. Build reference ITA distribution
    # --------------------------------------------------

    reference_distribution = (
        build_reference_distribution(
            reference_df
        )
    )

    print(
        "\nReference pixels:",
        len(reference_distribution),
    )

    print(
        "Reference median ITA:",
        np.median(
            reference_distribution
        ),
    )

    print(
        "Reference mean ITA:",
        np.mean(
            reference_distribution
        ),
    )

    print(
        "Reference ITA range:",
        np.min(reference_distribution),
        np.max(reference_distribution),
    )

    # --------------------------------------------------
    # 3. Save reference for reproducibility
    # --------------------------------------------------

    reference_dir = (
        dataset_root
        / "reference"
    )

    reference_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        reference_dir
        / "reference_ita.npy",
        reference_distribution,
    )

    reference_df.to_csv(
        reference_dir
        / "reference_images.csv",
        index=False,
    )

    # --------------------------------------------------
    # 4. Compute WD for each split
    # --------------------------------------------------

    splits = [
        (
            "train",
            df_train,
            train_path,
        ),
        (
            "valid",
            df_valid,
            valid_path,
        ),
        (
            "test",
            df_test,
            test_path,
        ),
    ]

    for (
        split_name,
        df,
        filepath,
    ) in splits:

        print(
            f"\nComputing {split_name} WD..."
        )

        df = add_wasserstein_distance(
            df,
            dataset_root,
            reference_distribution,
        )

        df.to_csv(
            filepath,
            index=False,
        )

        print(
            f"Saved: {filepath}"
        )

        print(
            df["wd"].describe()
        )


def main():

    for dataset_name in DATASETS:

        process_dataset(
            dataset_name
        )


if __name__ == "__main__":
    main()