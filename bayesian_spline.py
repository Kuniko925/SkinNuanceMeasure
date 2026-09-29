"""Binomial spline regression with a Gaussian (Laplace) posterior approximation.

For bin j: correct_j ~ Binomial(count_j, sigmoid(B(wd_j) @ beta)).
The proper Gaussian prior has precision
    I / coefficient_scale**2 + smoothness * D2.T @ D2,
where D2 takes second differences of adjacent spline coefficients.
Hyperparameters are fixed, not inferred. Credible intervals are pointwise,
conditional on the bins, knots, prior, and independent binomial trials.
Repeated images across seeds are not modeled as independent clusters here.
"""

import pickle

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer


class BayesianBinomialSpline:
    def __init__(self, n_knots=7, smoothness=10.0, coefficient_scale=5.0):
        if n_knots < 2 or smoothness < 0 or coefficient_scale <= 0:
            raise ValueError("Invalid spline/prior parameters.")
        self.n_knots = n_knots
        self.smoothness = smoothness
        self.coefficient_scale = coefficient_scale
        self.mu_value = None

    def fit(self, X, correct, count):
        X = np.asarray(X, dtype=float).reshape(-1, 1)
        correct = np.asarray(correct, dtype=float).reshape(-1)
        count = np.asarray(count, dtype=float).reshape(-1)
        if not (len(X) == len(correct) == len(count)):
            raise ValueError("WD, correct, and count must have equal lengths.")
        if not all(np.isfinite(v).all() for v in (X, correct, count)):
            raise ValueError("WD and binomial counts must be finite.")
        if (np.any(count <= 0) or np.any(correct < 0)
                or np.any(correct > count)
                or np.any(count != np.floor(count))
                or np.any(correct != np.floor(correct))):
            raise ValueError("Require integer counts with 0 <= correct <= count and count > 0.")
        n_unique = len(np.unique(X))
        if n_unique < 2:
            raise ValueError("At least two distinct WD values are required.")

        self.transformer_ = SplineTransformer(
            n_knots=min(self.n_knots, n_unique), degree=3,
            knots="uniform", include_bias=True, extrapolation="constant",
        )
        basis = self.transformer_.fit_transform(X)
        size = basis.shape[1]
        differences = np.diff(np.eye(size), n=2, axis=0)
        self.prior_precision_ = (
            np.eye(size) / self.coefficient_scale**2
            + self.smoothness * differences.T @ differences
        )

        def objective(beta):
            eta = basis @ beta
            value = np.sum(count * np.logaddexp(0, eta) - correct * eta)
            return value + 0.5 * beta @ self.prior_precision_ @ beta

        def gradient(beta):
            return (basis.T @ (count * expit(basis @ beta) - correct)
                    + self.prior_precision_ @ beta)

        def hessian(beta):
            p = expit(basis @ beta)
            weights = count * p * (1 - p)
            return basis.T @ (weights[:, None] * basis) + self.prior_precision_

        result = minimize(
            objective, np.zeros(size), jac=gradient, hess=hessian,
            method="trust-exact", options={"gtol": 1e-7, "maxiter": 500},
        )
        gradient_norm = float(np.linalg.norm(gradient(result.x), ord=np.inf))
        # Floating-point objective resolution can stop the trust-region solver
        # just above gtol; require a small residual even when it reports success.
        if not np.isfinite(result.fun) or gradient_norm > 1e-5:
            raise RuntimeError(f"Spline MAP fit did not converge: {result.message}")
        self.coef_ = result.x
        precision = hessian(self.coef_)
        np.linalg.cholesky(precision)
        self.covariance_ = np.linalg.solve(precision, np.eye(size))
        self.covariance_ = (self.covariance_ + self.covariance_.T) / 2
        self.wd_range_ = (float(X.min()), float(X.max()))
        self.diagnostics_ = {
            "method": "Laplace approximation at MAP",
            "gradient_norm": gradient_norm,
            "n_bins": len(X),
            "n_trials": int(count.sum()),
            "n_correct": int(correct.sum()),
        }
        return self

    def _latent_moments(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit the model before predicting.")
        X = np.asarray(X, dtype=float).reshape(-1, 1)
        if not np.isfinite(X).all():
            raise ValueError("Prediction WD values must be finite.")
        basis = self.transformer_.transform(X)
        mean = basis @ self.coef_
        variance = np.einsum("ij,jk,ik->i", basis, self.covariance_, basis)
        return mean, np.maximum(variance, 0)

    def predict(self, X):
        """Posterior mean probability, integrated by Gauss-Hermite quadrature."""
        mean, variance = self._latent_moments(X)
        nodes, weights = hermgauss(32)
        draws = mean[:, None] + np.sqrt(2 * variance[:, None]) * nodes
        return expit(draws) @ weights / np.sqrt(np.pi)

    def predict_interval(self, X, level=0.95):
        """Pointwise approximate credible interval for p, not future accuracy."""
        if not 0 < level < 1:
            raise ValueError("Credible level must be between 0 and 1.")
        mean, variance = self._latent_moments(X)
        radius = norm.ppf((1 + level) / 2) * np.sqrt(variance)
        return expit(mean - radius), expit(mean + radius)

    def estimate_mean(self, start, stop, num_points=1000):
        grid = np.linspace(start, stop, num_points)
        prediction = self.predict(grid)
        self.mu_value = float(prediction.mean())
        return grid, prediction

    def save(self, filepath, metadata=None):
        with open(filepath, "wb") as file:
            pickle.dump({
                "model_type": "bayesian_binomial_spline_laplace",
                # Compatibility with BRFairLoss, which calls brr_poly.predict.
                "brr_poly": self,
                "mu_value": self.mu_value,
                "metadata": metadata or {},
                "uncertainty_note": (
                    "Pointwise Laplace credible intervals; fixed knots/prior; "
                    "independent binomial trials. Repeated-image/seed dependence "
                    "is not modeled; intervals may be too narrow."
                ),
            }, file)
        print(f"Saved: {filepath}")
