import torch as t
import os
import numpy as np


def comparison(first,second,cond):
    if cond == 'min':
        return second < first
    elif cond == 'max':
        return second > first


class ModelsCheckpointer:
    def __init__(self, model, fn_pattern, fn_pattern_varnames, base_dir = './logs', replace = False,
                 watch_metric_names=None, watch_conditions = None):
        self.base_dir = base_dir
        self.model = model
        self.fn_pattern = fn_pattern
        self.fn_pattern_varnames = fn_pattern_varnames
        self.replace = replace
        self.prev_checkpoint_fname = None
        self.watch_metric_names = watch_metric_names
        self.watch_conditions = watch_conditions
        if watch_metric_names is not None:
            self.watch_metric_values = {}
            for m_name, m_cond in zip(watch_metric_names, watch_conditions):
                if m_cond == 'min':
                    self.watch_metric_values[m_name] = np.PINF
                elif m_cond == 'max':
                    self.watch_metric_values[m_name] = np.NINF
                else:
                    raise Exception('the condition should be either min or max, not %s' % str(m_cond))

    def save_models(self, pdict, metrics = None):
        if len(self.fn_pattern_varnames) > 0:
            fn = self.fn_pattern % tuple([pdict[name] for name in self.fn_pattern_varnames])
        else:
            fn = self.fn_pattern
        fn = os.path.join(self.base_dir, fn)

        if self.watch_metric_names is None:
            t.save(self.model.state_dict(), fn)
            if self.replace & (self.prev_checkpoint_fname is not None):
                os.remove(self.prev_checkpoint_fname)
            self.prev_checkpoint_fname = fn
        else:
            if metrics is None:
                raise Exception('you should pass metrics dictionary since you set checkpointing dependent on metrics values')
            for m_name, m_cond in zip(self.watch_metric_names, self.watch_conditions):
                curr_metric_value = self.watch_metric_values[m_name]
                new_metric_value = metrics[m_name]
                if comparison(curr_metric_value, new_metric_value, m_cond):
                    t.save(self.model.state_dict(), fn)
                    if self.replace & (self.prev_checkpoint_fname is not None):
                        os.remove(self.prev_checkpoint_fname)
                    self.prev_checkpoint_fname = fn
                    self.watch_metric_values[m_name] = new_metric_value