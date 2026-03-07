import numpy as np
import pandas as pd
from skincolor import IndividualTypologyAngle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent


class DistanceMeasure:
    def __init__(self, XX, YY):
        self.XX = XX
        self.YY = YY

    def probability_density_function(self, XX, YY):
        min_val = -90
        max_val = 90
        bins = np.linspace(min_val, max_val, min_val + max_val + 1)  # 180 degrees = from - 90 to 90 degees
        X_pdf, _ = np.histogram(XX, bins=bins, density=True)
        Y_pdf, _ = np.histogram(YY, bins=bins, density=True)

        return X_pdf, Y_pdf, bins

    def check_sign(self):
        pdf_X, pdf_Y, bins = self.probability_density_function(self.XX, self.YY)
        cdf_X = np.cumsum(pdf_X)
        cdf_Y = np.cumsum(pdf_Y)

        bins_center = (bins[:-1] + bins[1:]) / 2

        # 180 ITA angle
        median_X = np.interp(0.5, cdf_X / cdf_X[-1], bins_center)
        median_Y = np.interp(0.5, cdf_Y / cdf_Y[-1], bins_center)

        if median_X < median_Y:
            return 1
        else:
            return -1

    # Want to know positive/negative distibution shift
    def sign_wasserstein_distance(self):

        wd = self.Wasserstein_Dist(self.XX, self.YY)
        sign = self.check_sign()
        return wd * sign

    def Wasserstein_Dist(self):
        nx = len(self.XX)
        ny = len(self.YY)
        n = nx + ny

        XY = np.concatenate([self.XX, self.YY])
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]

        Res = 0
        E_CDF = 0
        F_CDF = 0
        power = 1

        for ii in range(0, n - 2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            height = abs(F_CDF - E_CDF)
            width = XY_Sorted[ii + 1] - XY_Sorted[ii]
            Res = Res + (height ** power) * width

        return Res

def main():

    roots = [PROJECT_ROOT / 'skin/UTKFace/', PROJECT_ROOT / 'skin/milk10k/clinical/',
             PROJECT_ROOT / 'skin/milk10k/dermoscopic/']
    for root in roots:
        train_path = root / 'df_train.csv'
        valid_path = root / 'df_valid.csv'
        test_path = root / 'df_test.csv'
        df_train = pd.read_csv(train_path, dtype={"label": str, "skin tone": str})
        df_valid = pd.read_csv(valid_path, dtype={"label": str, "skin tone": str})
        df_test = pd.read_csv(test_path, dtype={"label": str, "skin tone": str})
        dfs = [df_train, df_valid, df_test]
        paths = [train_path, valid_path, test_path]

        df_dark_skins = df_train.sort_values("angle").head(10)
        dark_skins = []
        for _, r in df_dark_skins.iterrows():
            ita = IndividualTypologyAngle(str(r['maskpath']))
            dark_skins.append(ita.compute_nuance_ita())
        dark_skin = np.concatenate(dark_skins)

        for i in range(len(dfs)):
            wd = []
            for _, r in dfs[i].iterrows():
                ita = IndividualTypologyAngle(str(r['maskpath']))
                skin = ita.compute_nuance_ita()
                dist = DistanceMeasure(dark_skin, skin)
                wd.append(dist.Wasserstein_Dist())
            dfs[i]['wd'] = wd
            dfs[i].to_csv(paths[i], index=False)

if __name__ == '__main__':
    main()