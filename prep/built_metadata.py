from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIN_ROOT = PROJECT_ROOT / "skin"

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
SKIN_TONE_GROUP_MAP = {
    "light": 0,
    "medium": 1,
    "dark": 2,
}

BINARY_LABEL_MAP = {
    "benign": 0,
    "malignant": 1,
}


# ============================================================
# Utility
# ============================================================

def normalize_isic_id(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".jpg", "", regex=False)
        .str.replace(".jpeg", "", regex=False)
        .str.replace(".png", "", regex=False)
    )


def find_image(
    dataset_name,
    isic_id,
    subdir,
):
    image_dir = (
        SKIN_ROOT
        / dataset_name
        / subdir
    )

    for ext in IMAGE_EXTENSIONS:
        path = image_dir / f"{isic_id}{ext}"

        if path.exists():
            return path

    return None


# ============================================================
# ITA computation
# ============================================================

def get_corbin_border_size(width,
                           percentage=0.04,
                           patch_width=8):
    return (
        int(
            math.ceil(
                width
                * percentage
                / patch_width
            )
        )
        * patch_width
    )


def compute_scalar_ita(
    skin_path,
    fov_path,
):

    # --------------------------------------------------------
    # Skin image
    # lesion area is black
    # --------------------------------------------------------

    skin_image = Image.open(
        skin_path
    ).convert("RGB")

    skin_rgb = np.asarray(
        skin_image,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # FOV image
    # outside dermoscopic field is black
    # --------------------------------------------------------

    fov_image = Image.open(
        fov_path
    ).convert("RGB")

    fov_rgb = np.asarray(
        fov_image,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Final healthy-skin region
    # --------------------------------------------------------

    skin_mask = np.any(
        skin_rgb != 0,
        axis=2
    )

    fov_mask = np.any(
        fov_rgb != 0,
        axis=2
    )

    valid_mask = (
        skin_mask
        & fov_mask
    )

    valid_count = int(
        valid_mask.sum()
    )

    valid_fraction = (
            valid_count
            / valid_mask.size
    )

    if valid_count == 0:
        return np.nan, 0, 0.0, np.nan, np.nan

    # --------------------------------------------------------
    # RGB -> CIELAB
    # --------------------------------------------------------

    lab = rgb2lab(
        skin_rgb
    )

    L = lab[:, :, 0][valid_mask]
    b = lab[:, :, 2][valid_mask]

    L = L[np.isfinite(L)]
    b = b[np.isfinite(b)]

    if len(L) == 0 or len(b) == 0:
        return np.nan, valid_count, np.nan, np.nan

    # --------------------------------------------------------
    # ±1 SD
    # --------------------------------------------------------

    L_mean = np.mean(L)
    L_std = np.std(L)

    b_mean = np.mean(b)
    b_std = np.std(b)

    if L_std > 0:
        L = L[
            (L >= L_mean - L_std)
            & (L <= L_mean + L_std)
        ]

    if b_std > 0:
        b = b[
            (b >= b_mean - b_std)
            & (b <= b_mean + b_std)
        ]

    if len(L) == 0 or len(b) == 0:
        return np.nan, valid_count, np.nan, np.nan

    median_L = float(
        np.median(L)
    )

    median_b = float(
        np.median(b)
    )

    # --------------------------------------------------------
    # ITA with atan2
    # --------------------------------------------------------

    ita = np.degrees(
        np.arctan2(
            median_L - 50.0,
            median_b
        )
    )

    return (
        float(ita),
        valid_count,
        float(valid_fraction),
        median_L,
        median_b,
    )

# ============================================================
# Add skin tone columns
# ============================================================

def add_skin_tone_information(df, dataset_name):

    ita_values = []
    ita_types = []
    skin_groups = []
    image_paths = []
    valid_pixels = []
    valid_fraction_values = []

    median_L_values = []
    median_b_values = []

    missing_images = []

    total = len(df)

    for i, row in df.iterrows():

        isic_id = str(
            row["isic_id"]
        ).strip()

        skin_path = find_image(
            dataset_name,
            isic_id,
            "skin",
        )

        fov_path = find_image(
            dataset_name,
            isic_id,
            "fov",
        )

        if (
                skin_path is None
                or fov_path is None
        ):
            ita_values.append(np.nan)
            ita_types.append(pd.NA)
            skin_groups.append(pd.NA)
            image_paths.append(pd.NA)
            valid_pixels.append(0)
            valid_fraction_values.append(0.0)
            median_L_values.append(np.nan)
            median_b_values.append(np.nan)

            missing_images.append(
                isic_id
            )

            continue

        (
            ita,
            n_pixels,
            valid_fraction,
            median_L,
            median_b,
        ) = compute_scalar_ita(
            skin_path,
            fov_path,
        )

        ita_type, group = ita_to_skin_tone(
            ita
        )

        ita_values.append(ita)
        ita_types.append(ita_type)
        skin_groups.append(group)

        # ITA calculation is based on this skin image
        image_paths.append(
            str(skin_path)
        )

        valid_pixels.append(
            n_pixels
        )

        valid_fraction_values.append(
            valid_fraction
        )

        median_L_values.append(
            median_L
        )

        median_b_values.append(
            median_b
        )

        if (i + 1) % 1000 == 0:
            print(
                f"{dataset_name}: "
                f"{i + 1}/{total}"
            )

    df["ita"] = ita_values

    df["ita_skin_type"] = pd.array(
        ita_types,
        dtype="Int64"
    )

    df["skin_tone_group"] = skin_groups
    df["skin_tone_group_id"] = pd.array(
        df["skin_tone_group"].map(SKIN_TONE_GROUP_MAP),
        dtype="Int64"
    )
    df["ita_valid_pixels"] = valid_pixels
    df["ita_valid_fraction"] = valid_fraction_values
    df["ita_filepath"] = image_paths

    df["ita_median_L"] = median_L_values
    df["ita_median_b"] = median_b_values

    print(
        f"\n{dataset_name} missing images:",
        len(missing_images)
    )

    if missing_images:
        print(
            "First missing IDs:",
            missing_images[:10]
        )

    return df

# ============================================================
# Diagnosis -> benign / malignant
# ============================================================

def classify_ham(dx):

    if pd.isna(dx):
        return pd.NA

    dx = str(
        dx
    ).strip().lower()

    malignant = {
        "mel",
        "bcc",
        "akiec",
    }

    benign = {
        "nv",
        "bkl",
        "df",
        "vasc",
    }

    if dx in malignant:
        return "malignant"

    if dx in benign:
        return "benign"

    return pd.NA


# ============================================================
# HAM10000
# ============================================================

def process_ham():

    path = (
        SKIN_ROOT
        / "HAM"
        / "HAM10000_metadata.csv"
    )

    df = pd.read_csv(
        path
    )

    print(
        "\n[HAM columns]"
    )
    print(
        df.columns.tolist()
    )

    # HAM uses image_id
    df["isic_id"] = normalize_isic_id(
        df["image_id"]
    )

    # --------------------------------------------------------
    # Benign / malignant
    # --------------------------------------------------------
    df["binary_label"] = (
        df["dx"]
        .apply(classify_ham)
    )

    df["binary_label_id"] = pd.array(
        df["binary_label"].map(BINARY_LABEL_MAP),
        dtype="Int64"
    )

    # --------------------------------------------------------
    # ITA skin tone
    # --------------------------------------------------------
    df = add_skin_tone_information(
        df,
        "HAM"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    save_path = (
        SKIN_ROOT
        / "HAM"
        / "data.csv"
    )

    df.to_csv(
        save_path,
        index=False
    )

    print(
        "\n================ HAM ================"
    )

    print(
        "Total:",
        len(df)
    )

    print(
        "\nBinary labels:"
    )
    print(
        df["binary_label"]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nITA 6-class distribution:"
    )
    print(
        df["ita_skin_type"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        "\nSkin-tone groups:"
    )
    print(
        df["skin_tone_group"]
        .value_counts(
            dropna=False
        )
    )

    print(
        f"\nSaved: {save_path}"
    )


# ============================================================
# BCN20000
# ============================================================

def process_bcn():

    path = (
        SKIN_ROOT
        / "BCN"
        / "metadata.csv"
    )

    df = pd.read_csv(path)

    print("\n[BCN columns]")
    print(df.columns.tolist())

    df["isic_id"] = normalize_isic_id(
        df["isic_id"]
    )

    # ========================================================
    # Binary label
    # ========================================================
    #
    # BCN metadata diagnosis_1
    #
    # Benign       -> benign
    # Malignant    -> malignant
    # Indeterminate / NaN -> excluded
    #

    diagnosis_map = {
        "Benign": "benign",
        "Malignant": "malignant",
    }

    df["binary_label"] = (
        df["diagnosis_1"]
        .map(diagnosis_map)
    )

    print("\nBefore exclusion:")
    print(
        df["binary_label"]
        .value_counts(dropna=False)
    )

    df = df[
        df["binary_label"].notna()
    ].copy()

    df = df.reset_index(drop=True)

    df["binary_label_id"] = pd.array(
        df["binary_label"].map(BINARY_LABEL_MAP),
        dtype="Int64"
    )

    # ========================================================
    # ITA categorical skin tone
    # ========================================================

    df = add_skin_tone_information(
        df,
        "BCN"
    )

    # ========================================================
    # Save
    # ========================================================

    save_path = (
        SKIN_ROOT
        / "BCN"
        / "data.csv"
    )

    df.to_csv(
        save_path,
        index=False
    )

    print(
        "\n================ BCN ================"
    )

    print(
        "Total:",
        len(df)
    )

    print("\nBinary labels:")
    print(
        df["binary_label"]
        .value_counts(dropna=False)
    )

    print("\nITA 6-class distribution:")
    print(
        df["ita_skin_type"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nSkin-tone groups:")
    print(
        df["skin_tone_group"]
        .value_counts(dropna=False)
    )

    print(f"\nSaved: {save_path}")

# ============================================================
# ITA -> 6 categories -> 3 groups
# ============================================================

def ita_to_skin_tone(ita):

    if pd.isna(ita):
        return pd.NA, pd.NA

    # Kinyanjui-based / Groh Eq. 14 thresholds
    if ita > 55:
        ita_type = 1

    elif ita > 41:
        ita_type = 2

    elif ita > 28:
        ita_type = 3

    elif ita > 19:
        ita_type = 4

    elif ita > 10:
        ita_type = 5

    else:
        ita_type = 6

    # 3-group setting
    if ita_type in [1, 2]:
        group = "light"

    elif ita_type in [3, 4]:
        group = "medium"

    else:
        group = "dark"

    return ita_type, group

# ============================================================
# Main
# ============================================================

def main():

    process_ham()
    process_bcn()

    import pandas as pd

    for db_name in ["HAM", "BCN"]:
        df = pd.read_csv(
            f"skin/{db_name}/data.csv"
        )

        print(
            f"\n===== {db_name} ====="
        )

        print("\nmedian L*:")
        print(
            df["ita_median_L"].describe(
                percentiles=[
                    .01, .05, .10, .25,
                    .50, .75, .90, .95, .99
                ]
            )
        )

        print("\nmedian b*:")
        print(
            df["ita_median_b"].describe(
                percentiles=[
                    .01, .05, .10, .25,
                    .50, .75, .90, .95, .99
                ]
            )
        )

        print(
            "\nb* <= 0:",
            (df["ita_median_b"] <= 0).sum()
        )

        print(
            "|b*| < 1:",
            (df["ita_median_b"].abs() < 1).sum()
        )

        print(
            "|b*| < 5:",
            (df["ita_median_b"].abs() < 5).sum()
        )

        print("\nITA type vs b* <= 0:")
        print(
            pd.crosstab(
                df["ita_skin_type"],
                df["ita_median_b"] <= 0
            )
        )


if __name__ == "__main__":
    main()