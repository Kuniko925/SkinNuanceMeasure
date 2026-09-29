import contextlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import bayesian
from bayesian_spline import BayesianBinomialSpline


class BinomialSplineTests(unittest.TestCase):
    def test_zero_and_perfect_accuracy_are_finite(self):
        x = np.linspace(0, 1, 25)
        for successes in (0, 40):
            model = BayesianBinomialSpline().fit(x, np.full(25, successes), np.full(25, 40))
            mean = model.predict(x)
            lower, upper = model.predict_interval(x)
            self.assertTrue(np.isfinite(mean).all())
            self.assertTrue(((0 < lower) & (lower < upper) & (upper < 1)).all())
            self.assertTrue((mean < .1).all() if successes == 0 else (mean > .9).all())

    def test_more_trials_narrow_intervals(self):
        x = np.linspace(0, 1, 25)
        widths = []
        for count in (20, 200):
            model = BayesianBinomialSpline().fit(x, np.full(25, count // 2), np.full(25, count))
            lower, upper = model.predict_interval(x)
            widths.append(np.mean(upper - lower))
            np.testing.assert_allclose(model.predict(x), .5, atol=1e-6)
        self.assertLess(widths[1], widths[0])

    def test_counts_weight_the_fit(self):
        x = np.linspace(0, 1, 10)
        # Conflicting observations at identical WD values: the larger bin wins.
        model = BayesianBinomialSpline().fit(
            np.tile(x, 2), np.r_[np.full(10, 90), np.ones(10)],
            np.r_[np.full(10, 100), np.full(10, 10)],
        )
        np.testing.assert_allclose(model.predict(x), 91 / 110, atol=.02)

    def test_invalid_counts_rejected(self):
        for correct, count in [([1, 3], [2, 2]), ([.5, 1], [2, 2]), ([0, 1], [0, 2])]:
            with self.assertRaises(ValueError):
                BayesianBinomialSpline().fit([0, 1], correct, count)

    def test_nonlinear_curve_and_serialization(self):
        x = np.linspace(0, 1, 25)
        truth = .5 + .3 * np.sin(2 * np.pi * x)
        model = BayesianBinomialSpline().fit(x, np.round(500 * truth), np.full(25, 500))
        self.assertLess(np.mean((model.predict(x) - truth) ** 2), .005)
        model.estimate_mean(0, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pkl'
            with contextlib.redirect_stdout(io.StringIO()):
                model.save(path)
            with path.open('rb') as file:
                saved = pickle.load(file)
        np.testing.assert_allclose(saved['brr_poly'].predict(x), model.predict(x))
        np.testing.assert_allclose(saved['brr_poly'].predict_interval(x), model.predict_interval(x))

    def test_pooled_validation_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / 'results'
            results.mkdir()
            for seed in (7, 29):
                frame = pd.DataFrame({
                    'wd': np.tile(np.linspace(0, 1, 30), 2),
                    'binary_label_id': np.repeat([0, 1], 30),
                    'pred': np.tile(np.arange(30) % 2, 2),
                })
                frame.to_csv(results / f'BCN_baseline_Test_seed{seed}_valid.csv', index=False)
            (results / 'BCN_baseline_Test_seed7_test.csv').write_text('not valid input')
            with patch.object(bayesian, 'PROJECT_ROOT', root), patch.object(bayesian, 'RESULTS_DIR', results), patch.object(bayesian, 'SEEDS', [7, 29]), contextlib.redirect_stdout(io.StringIO()):
                bayesian.process_run('BCN', 'baseline', 'Test')
                output = root / 'skin' / 'BCN' / 'bayes'
                self.assertEqual(len(list(output.glob('*.png'))), 1)
                self.assertEqual(len(list(output.glob('*.pkl'))), 2)
                for path in output.glob('*_bins.csv'):
                    bins = pd.read_csv(path)
                    self.assertEqual(bins['count'].sum(), 60)
                    self.assertEqual(bins['correct'].sum(), 30)
                for path in output.glob('*_curve.csv'):
                    curve = pd.read_csv(path)
                    self.assertTrue((curve.lower_95 < curve.upper_95).all())
                (results / 'BCN_baseline_Test_seed29_valid.csv').unlink()
                with self.assertRaises(FileNotFoundError):
                    bayesian.load_validation_results('BCN', 'baseline', 'Test')


if __name__ == '__main__':
    unittest.main()
