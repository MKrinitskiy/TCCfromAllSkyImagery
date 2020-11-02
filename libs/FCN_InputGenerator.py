try:
    import numpy as np
    from sklearn.model_selection import train_test_split
    import pickle
    import cv2
    import threading
    from .balanced_sampler import balanced_sampler
    from .targets_transformers import *
    import random
    from tqdm import tqdm
except ImportError as e:
    print(e)
    raise ImportError


def get_path_i(obj_count, batch_size):
    current_path_id = 0
    while True:
        cur_batch_indices = np.arange(current_path_id, current_path_id+batch_size)
        cur_batch_indices = np.mod(cur_batch_indices, obj_count)
        yield cur_batch_indices
        current_path_id = (current_path_id + batch_size) % obj_count



class FCN_InputGenerator:
    def __init__(self,
                 data_fname,
                 args,
                 batch_size=512,
                 subsetting_option=0.75,
                 model_type='OR',
                 rebalance=True,
                 debug=False,
                 **kwargs):
        print('loading files index...')

        if 'train_ds' in kwargs:
            self.dicts_per_date = kwargs['train_ds'].dicts_per_date
        else:
            with open(data_fname, 'rb') as f:
                self.dicts_per_date = pickle.load(f)

        # if 'train_ds' in kwargs:
        #     self.data_dicts = kwargs['train_ds'].data_dicts
        # with open(data_fname, 'rb') as f:
        #     self.data_dicts = pickle.load(f)

        self.batch_size = batch_size
        self.args = args
        self.model_type = model_type
        self.rebalance = rebalance
        self.debug = debug
        self.index = 0
        self.init_count = 0

        print('splitting data by dates...')
        dates = np.array([d for d in self.dicts_per_date])

        # if 'train_ds' in kwargs:
        #     self.dicts_per_date = kwargs['train_ds'].dicts_per_date
        # else:
        #     self.dicts_per_date = {}
        #     for curr_date in tqdm(dates):
        #         curr_date_dicts = [d for d in self.data_dicts if d['dt'].date() == curr_date]
        #         self.dicts_per_date[curr_date] = curr_date_dicts


        if isinstance(subsetting_option, float):
            self.curr_subset_dates, _ = train_test_split(dates, test_size=1 - subsetting_option)
        elif hasattr(subsetting_option, '__iter__'):
            self.curr_subset_dates = list(set(dates) - set([d for d in subsetting_option]))

        dicts = [self.dicts_per_date[d] for d in self.curr_subset_dates]
        self.dicts = [d for l in dicts for d in l]
        print('examples number: %d' % len(self.dicts))
        targets = np.array([d['observations_TCC'] for d in self.dicts]).astype(np.int32)

        print('balancing the data...')
        if self.rebalance:
            train_sample_size_per_class = np.max([tcc_class.sum() for tcc_class in [targets == c for c in np.unique(targets[targets < 8])]])
            self.dicts = balanced_sampler(X=self.dicts, y=targets, sample_size_per_class=train_sample_size_per_class)
            print('rebalanced examples number: %d' % len(self.dicts))

        self.varnames = ['r_perc', 'g_perc', 'b_perc', 'h_perc', 's_perc', 'v_perc']

        self.lock = threading.Lock()  # mutex for input path
        self.indices_gen = get_path_i(len(self.dicts), batch_size=batch_size)
        self.transformer = targets_transformer_np(model_type=self.model_type, batch_size=len(self.dicts))


    @property
    def dates_used(self):
        return self.curr_subset_dates

    def get_samples_count(self):
        """ Returns the total number of images needed to train an epoch """
        return len(self.dicts)

    def get_batches_count(self):
        """ Returns the total number of batches needed to train an epoch """
        return int(self.get_samples_count() / self.batch_size)

    def shuffle(self):
        print('shuffling source data...')
        random.shuffle(self.dicts)
        print('transforming source data...')
        self.data = np.concatenate([np.concatenate([self.dicts[idx][k] for k in self.varnames])[np.newaxis, :] for idx in range(len(self.dicts))], axis=0)
        self.targets = np.array([int(self.dicts[idx]['observations_TCC']) for idx in range(len(self.dicts))])
        self.targets = self.transformer(self.targets)


    def __iter__(self):
        while True:
            with self.lock:
                if (self.init_count == 0):
                    self.shuffle()
                    self.init_count = 1
                batch_indices = next(self.indices_gen)
                x = self.data[batch_indices]
                y = self.targets[batch_indices]
                yield (np.float32(x), np.float32(y))
