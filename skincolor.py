from PIL import Image
from skimage import color
from derm_ita import derm_ita
import numpy as np

class IndividualTypologyAngle:
    def __init__(self, filepath):
        self.RGB = Image.open(filepath)
        self.CIELAB = np.array(color.rgb2lab(self.RGB))

    def compute_mean_ita(self):
        # Calculate mean ITA value
        ita = derm_ita.get_ita(self.RGB)
        return ita

    def compute_nuance_ita(self):
        # based on https://github.com/AdamCorbinFAUPhD/derm_ita/blob/master/derm_ita/ita_core_computations.py
        L = self.CIELAB[:, :, 0]
        L = np.where(L != 0, L, np.nan)

        B = self.CIELAB[:, :, 2]
        B = np.where(B != 0, B, np.nan)

        # If both nan, remove the pixel
        nan_indices = np.isnan(L) & np.isnan(B)
        L = L[~nan_indices]
        B = B[~nan_indices]

        # Calculate ITA Distribution
        ita = np.arctan2(L - 50, B) * (180 / np.pi)
        return ita[~np.isnan(ita)]


