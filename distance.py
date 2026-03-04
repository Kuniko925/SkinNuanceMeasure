import numpy as np


class DistanceMeasure:
    def __init__(self, XX, YY):
        self.XX = XX
        self.YY = YY

    def probability_density_function(self, XX, YY):
        min_val = -90
        max_val = 90
        bins = np.linspace(min_val, max_val, 180 + 1)  # 180 degrees = from - 90 to 90 degees

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

    def Wasserstein_Dist(self, XX, YY):
        nx = len(XX)
        ny = len(YY)
        n = nx + ny

        XY = np.concatenate([XX, YY])
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
            Res = Res + (height ** power) * width;

        return Res