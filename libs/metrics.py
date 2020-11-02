import torch as t
from .targets_transformers import *


class accuracy:
    def __init__(self, name: str = 'accuracy', model_type: str = 'PC', batch_size: int = 16, classes_number: int = 9, **kwargs):
        self.model_type = model_type
        self.batch_size = batch_size
        self.name = name
        self.classes_number = classes_number
        self.target_transformer = targets_transformer(self.model_type, self.batch_size, classes_number=self.classes_number)

    def __call__(self, y_pred, y_true):
        if y_pred.dim() == 1:
            y_pred = t.reshape(y_pred, (1,-1))
        if y_true.dim() == 1:
            y_true = t.reshape(y_true, (1,-1))
        true_cls = self.target_transformer.transform_back(y_true)
        pred_cls = self.target_transformer.transform_back(y_pred)
        eq = t.squeeze(true_cls) == t.squeeze(pred_cls)
        eq = eq.float()
        if eq.dim() == 0:
            acc = eq
        else:
            acc = eq.sum()/eq.shape[0]
        return acc


#TODO: fix the metric (does not work properly in case of ordinal encoding) - DONE
class diff_leq_accuracy:
    def __init__(self, name: str = 'diff_leq_accuracy', model_type: str = 'PC', batch_size:int = 16, leq_threshold:int = 1, classes_number: int = 9, **kwargs):
        self.model_type = model_type
        self.batch_size = batch_size
        self.leq_threshold = leq_threshold
        self.name = name
        self.classes_number = classes_number
        self.target_transformer = targets_transformer(self.model_type, self.batch_size, classes_number=self.classes_number)


    def __call__(self, y_pred, y_true):
        if y_pred.dim() == 1:
            y_pred = t.reshape(y_pred, (1,-1))
        if y_true.dim() == 1:
            y_true = t.reshape(y_true, (1,-1))
        true_cls = self.target_transformer.transform_back(y_true)
        pred_cls = self.target_transformer.transform_back(y_pred)
        diff = t.squeeze(true_cls) - t.squeeze(pred_cls)
        diff = t.abs(diff.float())
        leq_sum = (diff <= self.leq_threshold).float().sum()
        acc_leq = leq_sum/diff.shape[0]
        return acc_leq


#TODO: MSE metric for TCC calculated as classification and ordinal random var. (if necessary)

#TODO: target interpretation and accuracy estimation in case of ordinal encoding - DONE