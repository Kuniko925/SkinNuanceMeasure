from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import cv2
import numpy as np
from config import DATASETS, IMAGE_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def make_fov_mask(rgb):

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Detect non-background region
    _, mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # Largest connected component
    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if n_labels <= 1:
        return mask.astype(bool)

    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    fov_mask = (
        labels == largest
    ).astype(np.uint8)

    # -----------------------------------------
    # Remove dark/vignetted FOV boundary
    # -----------------------------------------

    distance = cv2.distanceTransform(
        fov_mask,
        cv2.DIST_L2,
        5
    )

    h, w = fov_mask.shape

    # about 4% of image size
    margin = max(
        5,
        int(round(min(h, w) * 0.04))
    )

    fov_mask = (
        distance >= margin
    )

    return fov_mask

def process_image(
    input_path,
    output_path,
):

    # OpenCV loads BGR
    bgr = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR
    )

    if bgr is None:
        print(
            f"Could not read: {input_path}"
        )
        return False

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )

    # Compute FOV
    fov_mask = make_fov_mask(
        rgb
    )

    # Outside FOV = pure black
    output_rgb = np.zeros_like(
        rgb
    )

    output_rgb[fov_mask] = (
        rgb[fov_mask]
    )

    output_bgr = cv2.cvtColor(
        output_rgb,
        cv2.COLOR_RGB2BGR
    )

    # Always save PNG
    success = cv2.imwrite(
        str(output_path),
        output_bgr
    )

    if not success:
        print(
            f"Could not save: {output_path}"
        )

    return success


def process_dataset(
    dataset_name
):

    dataset_root = (
        PROJECT_ROOT
        / "skin"
        / dataset_name
    )

    input_dir = (
        dataset_root
        / "data"
    )

    output_dir = (
        dataset_root
        / "fov"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = sorted(
        [
            p
            for p in input_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    total = len(image_paths)

    print(
        "\n"
        "====================================\n"
        f"{dataset_name}\n"
        "===================================="
    )

    print(
        f"Input : {input_dir}"
    )

    print(
        f"Output: {output_dir}"
    )

    print(
        f"Images: {total}"
    )

    success_count = 0

    for i, input_path in enumerate(
        image_paths,
        start=1
    ):

        # Always output PNG
        output_path = (
            output_dir
            / f"{input_path.stem}.png"
        )

        success = process_image(
            input_path,
            output_path,
        )

        if success:
            success_count += 1

        if (
            i % 500 == 0
            or i == total
        ):
            print(
                f"{i}/{total}"
            )

    print(
        f"Completed: "
        f"{success_count}/{total}"
    )


def main():

    for dataset_name in DATASETS:
        process_dataset(
            dataset_name
        )


if __name__ == "__main__":
    main()