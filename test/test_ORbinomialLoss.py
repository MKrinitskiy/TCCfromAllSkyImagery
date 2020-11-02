from unittest import TestCase
from libs.ORbinomialLoss import ORbinomialLoss
import numpy as np
from libs.targets_transformers import targets_transformer
from scipy.optimize import minimize
from .binomial_proba_calculator_np import binomial_proba_calculator



def test_calc():
    bs_values = [1] + [s for s in np.random.randint(2, 128, 32)]
    K_values = [s for s in np.random.randint(3, 32, 32)]
    for bs in bs_values:
        for K in K_values:
            transformer = targets_transformer(model_type='PC', batch_size=bs, classes_number=K)
            lobj = ORbinomialLoss(classes_number=K)
            # generate y_true
            y_true = np.random.choice(np.arange(0, K, 1), bs)
            y_true = transformer(y_true)

            px = np.random.rand(bs, 1).astype(np.float32)
            loss = lobj(px, y_true)
            assert loss >= 0.0


def test_perfect_distr():
    bs_values = [1] + [s for s in np.random.randint(2, 128, 8)]
    K_values = [s for s in np.random.randint(3, 16, 8)]
    for bs in bs_values:
        for K in K_values:
            transformer = targets_transformer(model_type='PC', batch_size=bs, classes_number=K)
            lobj = ORbinomialLoss(classes_number=K)
            # generate y_true
            y_true = np.random.choice(np.arange(0, K, 1), bs)
            y_true_oh = transformer(y_true)

            bpcalc = binomial_proba_calculator(classes_number=K)
            optimal_px = np.zeros((bs,))
            for idx, k in enumerate(y_true):
                if k == 0:
                    curr_px = 0.0
                elif k == K-1:
                    curr_px = 1.0
                else:
                    curr_px = minimize(lambda x: (1.0 - bpcalc(x))[k], x0=np.array([0.5]), bounds=[(0.0, 1.0)]).x[0]
                optimal_px[idx] = curr_px

            optimal_loss = float(lobj(optimal_px[:, np.newaxis].astype(np.float32), y_true_oh).cpu().numpy())
            suboptimal_px = [optimal_px + np.random.randn(*(optimal_px.shape)) / 100.0 for i in range(100)]
            suboptimal_px = np.clip(suboptimal_px, 0.0, 1.0)
            suboptimal_losses = [float(lobj(px[:,np.newaxis].astype(np.float32), y_true_oh).cpu().numpy()) for px in suboptimal_px]

            assert np.abs(np.clip(np.array([(sul - optimal_loss) / optimal_loss for sul in suboptimal_losses]), -np.inf, 0.0)).max() <= 0.1
