import numpy as np
import torch as t
import numbers
from .elastic_transform.gather_nd import *


def __consistent_args(inp, condition, indices):
    assert len(inp.shape) == 2, 'only works for batch x dim tensors along the dim axis'
    mask = condition(inp).float() * indices.unsqueeze(0).expand_as(inp)
    return t.argmax(mask, dim=1)

def consistent_find_leftmost(inp, condition):
    indices = t.arange(inp.size(1), 0, -1, dtype=t.float, device=inp.device)
    return __consistent_args(inp, condition, indices)

def consistent_find_rightmost(inp, condition):
    indices = t.arange(0, inp.size(1), 1, dtype=t.float, device=inp.device)
    return __consistent_args(inp, condition, indices)


class targets_transformer:
    def __init__(self, model_type, batch_size, classes_number = 9):
        self.proba_threshold = 0.5
        self.model_type = model_type
        self.batch_size = batch_size
        self.classes_number = classes_number
        self.classes = t.arange(0, classes_number, 1, dtype=t.long).cuda()
        classes_t = self.classes.view((1,-1))
        self.targets_oh_blank = t.repeat_interleave(classes_t, self.batch_size, 0).cuda()

    def transform_back(self, y_pred):
        if y_pred.dim() == 1:
            y_pred = t.reshape(y_pred, (1,-1))

        if self.model_type in ['PC', 'ORbin']:
            classes_pred = t.argmax(y_pred, dim=-1)
            classes_pred = t.gather(self.classes, 0, classes_pred)
            return classes_pred
        elif self.model_type == 'OR':
            y_pred_gtt = (y_pred > self.proba_threshold)
            all_ones_pred = t.all(y_pred_gtt, dim=1)
            class_index_pred = consistent_find_leftmost(y_pred_gtt.long(), lambda x: x <= self.proba_threshold)
            class_index_pred = (class_index_pred - 1) * (~all_ones_pred).long() + all_ones_pred.long() * (self.classes_number-1)
            class_index_pred = t.clamp(class_index_pred, 0, self.classes_number-1)
            classes_pred = t.gather(self.classes, 0, class_index_pred)
            return classes_pred

    def __call__(self, targets, *args, **kwargs):
        if type(targets) == np.ndarray:
            targets = t.from_numpy(targets).long().cuda()
        elif t.is_tensor(targets):
            targets = targets.long()
            if not targets.is_cuda:
                targets = targets.cuda()
        targets = targets.view((-1,1)) # B x 1

        if self.model_type in ['PC', 'ORbin']:
            targets = t.repeat_interleave(targets, self.classes_number, -1)  # B x 9
            targets_oh = (targets == self.targets_oh_blank).long()
            return t.reshape(targets_oh.squeeze(), (self.batch_size, -1))
        elif self.model_type == 'OR':
            leq = (targets >= self.targets_oh_blank).long()
            return t.reshape(leq.squeeze(), (self.batch_size, -1))



class targets_transformer_np:
    def __init__(self, model_type, batch_size, classes_number = 9):
        self.proba_threshold = 0.5
        self.model_type = model_type
        self.batch_size = batch_size
        self.classes_num = classes_number
        self.classes = np.arange(0, self.classes_num, 1, dtype=np.int32)
        classes_t = self.classes.reshape((-1, self.classes_num))
        self.targets_oh_blank = np.tile(classes_t, (self.batch_size, 1))

    def transform_back(self, y_pred):
        if y_pred.ndim == 1:
            y_pred = np.reshape(y_pred, (1,-1))

        if self.model_type in ['PC', 'ORbin']:
            classes_pred = np.argmax(y_pred, axis=-1)
            classes_pred = self.classes[classes_pred]
            return classes_pred
        elif self.model_type == 'OR':
            y_pred_gtt = (y_pred > self.proba_threshold)
            all_ones_pred = np.all(y_pred_gtt, axis=1)
            class_index_pred = np.argmin(y_pred_gtt.astype(np.float32), axis=-1)
            class_index_pred = (class_index_pred-1)*np.invert(all_ones_pred).astype(np.int32) + all_ones_pred.astype(np.int32)*(self.classes_num-1)
            class_index_pred = np.clip(class_index_pred, 0, self.classes_num-1)
            classes_pred = self.classes[class_index_pred]
            return classes_pred

    def __call__(self, targets, *args, **kwargs):
        if isinstance(targets, numbers.Number):
            targets = np.array([[targets]])
        elif isinstance(targets, list):
            targets = np.array(targets).reshape((-1,1))
        elif isinstance(targets, np.ndarray):
            targets = targets.reshape((-1, 1))
        targets = np.tile(targets, (1, self.classes_num))

        if self.model_type in ['PC', 'ORbin']:
            targets_oh = (targets == self.targets_oh_blank).astype(np.float32)
            return targets_oh.squeeze().reshape((self.batch_size, -1))
        elif self.model_type == 'OR':
            leq = (targets >= self.targets_oh_blank).astype(np.float32)
            return leq.squeeze().reshape((self.batch_size, -1))
