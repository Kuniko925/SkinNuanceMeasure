import unittest

import numpy as np
import torch

from bayesian_regularization import BRFairLoss
from train import ModelTrainer


class FixedRegression:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict(self, X):
        return self.probabilities[:len(X)]


class BRFairLossTests(unittest.TestCase):
    def make_loss(self, class_zero, class_one):
        return BRFairLoss({
            0: {"brr_poly": FixedRegression(class_zero)},
            1: {"brr_poly": FixedRegression(class_one)},
        })

    def test_normalized_weights_and_gradient(self):
        ce = torch.tensor([.2, 1., 2.], dtype=torch.float64, requires_grad=True)
        loss = self.make_loss([.9, .7], [.5])(
            ce, torch.tensor([10.5, 20.5, 30.5]), torch.tensor([0, 0, 1]),
        )
        # Class 0: (.1*.2 + .3*1)/.4 = .8; class 1: 2; mean = 1.4.
        self.assertAlmostEqual(loss.item(), 1.4, places=6)
        loss.backward()
        torch.testing.assert_close(
            ce.grad, torch.tensor([.125, .375, .5], dtype=torch.float64),
        )

    def test_uniform_weights_cancel_within_class(self):
        ce = torch.tensor([1., 3., 5.])
        labels = torch.tensor([0, 0, 1])
        for probability in (.1, .9):
            loss = self.make_loss([probability] * 2, [probability])(ce, ce, labels)
            self.assertAlmostEqual(loss.item(), 3.5, places=5)

    def test_zero_weights_and_empty_batch(self):
        ce = torch.tensor([2.], requires_grad=True)
        loss = self.make_loss([1.], [])(ce, ce, torch.tensor([0]))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(ce.grad.item(), 0.)
        empty = torch.empty(0, dtype=torch.float64, requires_grad=True)
        loss = self.make_loss([], [])(empty, empty, torch.empty(0, dtype=torch.long))
        self.assertEqual(loss.dtype, empty.dtype)
        loss.backward()

    def test_ce_class_weights_only_removed_for_individual(self):
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.class_weights = torch.tensor([.6, 2.5])
        for mode in ['baseline', 'reweight', 'regularization', 'individual']:
            trainer.train_method = mode
            criterion = trainer._make_classification_criterion()
            self.assertEqual(criterion.reduction, 'mean' if mode == 'baseline' else 'none')
            if mode == 'individual':
                self.assertIsNone(criterion.weight)
            else:
                torch.testing.assert_close(criterion.weight, trainer.class_weights)


if __name__ == '__main__':
    unittest.main()
