import torch as t
import numpy as np
from .targets_transformers import targets_transformer



def dims(a):
    if type(a) == np.ndarray:
        return a.ndim
    elif t.is_tensor(a):
        return a.dim()



class ORbinomialLoss:
    def __init__(self, name:str = 'ORbinomialLoss', classes_number:int = 9, **kwargs):
        # self.batch_size = batch_size
        self.name = name
        self.K = classes_number
        mult_factors = np.array([np.math.factorial(self.K - 1) / np.math.factorial(k - 1) / np.math.factorial(self.K - k) for k in range(1, self.K + 1)])
        self.mult_factors = t.from_numpy(mult_factors).cuda()
        k = np.arange(1, self.K + 1, 1)
        self.k = t.from_numpy(k).float().cuda()
        # self.one_hot_transformer = targets_transformer(model_type='PC', batch_size=batch_size, classes_number=classes_number)

    def calc(self, y_pred, y_true):
        return self.__call__(y_pred, y_true)

    def __call__(self, y_pred, y_true):
        # y_pred should contain px values per object - so it should be of shape (batch_size, 1)
        # assert list(y_pred.shape) == [self.batch_size, 1]

        if dims(y_pred) > 1:
            y_pred = y_pred.reshape((-1,))
        if dims(y_true) == 1:
            y_true = y_true.reshape((1,-1))
        if type(y_pred) == np.ndarray:
            y_pred = t.from_numpy(y_pred).cuda()
        if type(y_true) == np.ndarray:
            y_true = t.from_numpy(y_true).cuda()

        # y_true should be one-ho encoded
        assert t.allclose(t.sum(y_true, dim=1), t.ones_like(t.sum(y_true, dim=1)))
        # classes number of y_true should be K
        assert y_true.shape[1] == self.K
        assert y_true.shape[0] == y_pred.shape[0]


        y_pred_mesh, k_mesh = t.meshgrid(y_pred, self.k)
        pow1 = t.pow(y_pred_mesh, k_mesh - 1)
        pow2 = t.pow((1 - y_pred_mesh), (self.K - self.k))
        mult_factors_mesh = t.repeat_interleave(t.unsqueeze(self.mult_factors, 0), y_true.shape[0], 0)
        classes_probs = mult_factors_mesh * pow1 * pow2
        deltas = (1 - y_true).float()
        loss = t.mean(t.square(classes_probs - deltas))

        return loss
