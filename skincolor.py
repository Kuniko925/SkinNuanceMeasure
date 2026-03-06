from PIL import Image
from skimage import color, io
from derm_ita import derm_ita
import numpy as np
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent

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

def main():

    roots = [PROJECT_ROOT / 'skin/UTKFace/', PROJECT_ROOT / 'skin/milk10k/clinical/', PROJECT_ROOT / 'skin/milk10k/dermoscopic/']
    for root in roots:
        df_train = pd.read_csv(f"{root}/df_train.csv", dtype={"label": str, "skin tone": str})
        df_valid = pd.read_csv(f"{root}/df_valid.csv", dtype={"label": str, "skin tone": str})
        df_test = pd.read_csv(f"{root}/df_test.csv", dtype={"label": str, "skin tone": str})
        dfs = {'train':df_train, 'valid':df_valid, 'test':df_test}

        for key, df in dfs.items():
            angles = []
            for _, r in df.iterrows():
                maskpath = PROJECT_ROOT / r["maskpath"]
                ita = IndividualTypologyAngle(maskpath)
                angle = ita.compute_mean_ita()
                angles.append(angle)
                if np.isnan(angle):
                    print("empty/invalid:", maskpath)
            df["angle"] = angles
            df = df.dropna(subset=["angle"])
            df.to_csv(f"{root}/df_{key}.csv", index=False)
            print(f'Completed {key}.')

if __name__ == "__main__":
    main()