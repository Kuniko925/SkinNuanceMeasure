from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from config import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIN_ROOT = PROJECT_ROOT / "skin"
SPLIT_SEED = 42
MIN_VALID_FRACTION = 0.15

def prepare_group_id(df):
    group_id = df["lesion_id"].astype("string").copy()

    missing = group_id.isna()

    group_id.loc[missing] = (
        "single_"
        + df.loc[missing, "isic_id"].astype(str)
    )

    return group_id

def find_original_image(
    dataset_name,
    isic_id,
):
    image_dir = (
        SKIN_ROOT
        / dataset_name
        / "data"
    )

    for ext in [
        ".jpg",
        ".jpeg",
        ".png",
    ]:
        path = (
            image_dir
            / f"{isic_id}{ext}"
        )

        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Original image not found: "
        f"{dataset_name}/{isic_id}"
    )

def split_dataset(dataset_name):

    data_path = (
        SKIN_ROOT
        / dataset_name
        / "data.csv"
    )

    df = pd.read_csv(data_path)

    before = len(df)

    df = df[
        df["binary_label"].notna()
        & df["skin_tone_group"].notna()
        & df["skin_tone_group_id"].notna()
        & df["ita"].notna()
        ].copy()



    df = df[
        df["ita_valid_fraction"]
        >= MIN_VALID_FRACTION
        ].copy()

    df = df.reset_index(drop=True)

    print(
        f"Excluded invalid skin-tone samples: "
        f"{before - len(df)}"
    )

    # Classification image path
    df["filepath"] = df[
        "isic_id"
    ].apply(
        lambda x: find_original_image(
            dataset_name,
            str(x),
        )
    )
    print(
        f"\n================ {dataset_name} ================"
    )
    print("Total:", len(df))

    required_columns = [
        "binary_label_id",
        "skin_tone_group_id",
        "lesion_id",
        "isic_id",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"{dataset_name}: missing column '{col}'"
            )

    df = df[
        df["binary_label_id"].notna()
        & df["skin_tone_group_id"].notna()
    ].copy()

    df["binary_label_id"] = (
        df["binary_label_id"]
        .astype(int)
    )

    df["skin_tone_group_id"] = (
        df["skin_tone_group_id"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Stratification:
    # diagnosis × skin tone
    #
    #
    # 0_0 = benign/light
    # 0_1 = benign/medium
    # 0_2 = benign/dark
    # 1_0 = malignant/light
    # ...
    # --------------------------------------------------------
    df["stratify_key"] = (
        df["binary_label_id"].astype(str)
        + "_"
        + df["skin_tone_group_id"].astype(str)
    )

    df["_group_id"] = prepare_group_id(df)

    # ========================================================
    # TEST: 20%
    # ========================================================

    sgkf_test = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPLIT_SEED,
    )

    trainval_idx, test_idx = next(
        sgkf_test.split(
            X=df,
            y=df["stratify_key"],
            groups=df["_group_id"],
        )
    )

    df_trainval = (
        df.iloc[trainval_idx]
        .copy()
        .reset_index(drop=True)
    )

    df_test = (
        df.iloc[test_idx]
        .copy()
        .reset_index(drop=True)
    )

    sgkf_val = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPLIT_SEED + 1,
    )

    train_idx, valid_idx = next(
        sgkf_val.split(
            X=df_trainval,
            y=df_trainval["stratify_key"],
            groups=df_trainval["_group_id"],
        )
    )

    df_train = (
        df_trainval.iloc[train_idx]
        .copy()
        .reset_index(drop=True)
    )

    df_valid = (
        df_trainval.iloc[valid_idx]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # Leakage check
    # ========================================================

    train_lesions = set(df_train["_group_id"])
    valid_lesions = set(df_valid["_group_id"])
    test_lesions = set(df_test["_group_id"])

    assert train_lesions.isdisjoint(valid_lesions)
    assert train_lesions.isdisjoint(test_lesions)
    assert valid_lesions.isdisjoint(test_lesions)

    df_train["label"] = df_train["binary_label_id"]
    df_valid["label"] = df_valid["binary_label_id"]
    df_test["label"] = df_test["binary_label_id"]

    df_train["group"] = df_train["skin_tone_group_id"]
    df_valid["group"] = df_valid["skin_tone_group_id"]
    df_test["group"] = df_test["skin_tone_group_id"]

    drop_cols = [
        "stratify_key",
        "_group_id",
    ]

    df_train = df_train.drop(
        columns=drop_cols
    )

    df_valid = df_valid.drop(
        columns=drop_cols
    )

    df_test = df_test.drop(
        columns=drop_cols
    )

    # ========================================================
    # Save
    # ========================================================

    root = (
        SKIN_ROOT
        / dataset_name
    )

    train_path = root / "df_train.csv"
    valid_path = root / "df_valid.csv"
    test_path = root / "df_test.csv"

    df_train.to_csv(
        train_path,
        index=False
    )

    df_valid.to_csv(
        valid_path,
        index=False
    )

    df_test.to_csv(
        test_path,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    total = len(df)

    print(
        f"Train: {len(df_train)} "
        f"({len(df_train) / total:.1%})"
    )

    print(
        f"Valid: {len(df_valid)} "
        f"({len(df_valid) / total:.1%})"
    )

    print(
        f"Test : {len(df_test)} "
        f"({len(df_test) / total:.1%})"
    )

    for name, split_df in [
        ("TRAIN", df_train),
        ("VALID", df_valid),
        ("TEST", df_test),
    ]:

        print(
            f"\n--- {name}: diagnosis ---"
        )
        print(
            split_df[
                "binary_label"
            ].value_counts()
        )

        print(
            f"\n--- {name}: skin tone ---"
        )
        print(
            split_df[
                "skin_tone_group"
            ].value_counts()
        )

    print(
        "\nNo lesion-level leakage detected."
    )

    print(
        f"Saved:\n"
        f"{train_path}\n"
        f"{valid_path}\n"
        f"{test_path}"
    )


def main():

    for dataset_name in DATASETS:
        split_dataset(
            dataset_name
        )

    # ========================================================
    # Final cohort summary
    # ========================================================

    for db_name in DATASETS:

        root = (
            SKIN_ROOT
            / db_name
        )

        df_train = pd.read_csv(
            root / "df_train.csv"
        )

        df_valid = pd.read_csv(
            root / "df_valid.csv"
        )

        df_test = pd.read_csv(
            root / "df_test.csv"
        )

        df = pd.concat(
            [
                df_train,
                df_valid,
                df_test,
            ],
            ignore_index=True,
        )

        print(
            f"\n"
            f"====================================\n"
            f"{db_name} FINAL COHORT\n"
            f"===================================="
        )

        print(
            "Total:",
            len(df)
        )

        print(
            "\nDiagnosis:"
        )

        print(
            df["binary_label"]
            .value_counts(
                dropna=False
            )
        )

        print(
            "\nSkin tone:"
        )

        print(
            df["skin_tone_group"]
            .value_counts(
                dropna=False
            )
        )

        print(
            "\nValid skin fraction:"
        )

        print(
            df[
                "ita_valid_fraction"
            ].describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
        )

        print(
            "\nITA:"
        )

        print(
            df["ita"].describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
        )

if __name__ == "__main__":
    main()