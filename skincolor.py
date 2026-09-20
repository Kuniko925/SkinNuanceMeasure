from pathlib import Path
PROJECT_ROOT = Path(__file__).parent

from PIL import Image
from skimage import color
import numpy as np


class IndividualTypologyAngle:

    def __init__(
        self,
        skin_filepath,
        fov_filepath,
    ):
        self.skin_rgb = np.asarray(
            Image.open(
                skin_filepath
            ).convert("RGB")
        )

        self.fov_rgb = np.asarray(
            Image.open(
                fov_filepath
            ).convert("RGB")
        )

        if (
            self.skin_rgb.shape
            != self.fov_rgb.shape
        ):
            raise ValueError(
                f"Shape mismatch: "
                f"{skin_filepath} vs "
                f"{fov_filepath}"
            )

        self.CIELAB = color.rgb2lab(
            self.skin_rgb
        )

    def get_valid_mask(self):

        # lesion-excluded region
        skin_mask = np.any(
            self.skin_rgb != 0,
            axis=2
        )

        # valid dermoscopic FOV
        fov_mask = np.any(
            self.fov_rgb != 0,
            axis=2
        )

        return (
            skin_mask
            & fov_mask
        )

    def compute_nuance_ita(self):

        L = self.CIELAB[:, :, 0]
        B = self.CIELAB[:, :, 2]

        valid = (
            self.get_valid_mask()
            & np.isfinite(L)
            & np.isfinite(B)
        )

        L = L[valid]
        B = B[valid]

        if len(L) == 0:
            return np.array([])

        # Pixel-wise ITA with atan2
        ita = np.degrees(
            np.arctan2(
                L - 50.0,
                B
            )
        )

        return ita[
            np.isfinite(ita)
        ]