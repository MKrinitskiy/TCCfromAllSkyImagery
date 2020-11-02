import os
from .service_defs import DoesPathExistAndIsFile, DoesPathExistAndIsDirectory
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from .targets_transformers import *
from torch.utils.data import DataLoader, Dataset
import torch as t
import cv2
from multiprocessing import Pool


def batches_generator(items, batch_size=16, shuffle = True):
    elements = len(items)
    if shuffle:
        np.random.shuffle(items)
    batches = elements // batch_size
    if batches * batch_size < elements:
        batches += 1

    while True:
        for b_idx in range(batches):
            if b_idx < batches - 1:
                curr_batch = items[b_idx * batch_size:(b_idx + 1) * batch_size]
            else:
                curr_batch = items[b_idx * batch_size:]
            yield curr_batch
        if shuffle:
            np.random.shuffle(items)



class SIAdataset(Dataset):
    def __init__(self, data_index_fname,
                 img_size = (3,256,256),
                 batch_size = 16,
                 subsetting_option = 0.75,
                 model_type = 'OR',
                 augment=True,
                 rebalance = True,
                 debug = False):
        super(SIAdataset, self).__init__()

        assert DoesPathExistAndIsFile(data_index_fname)
        self.data_index_fname = data_index_fname
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.model_type = model_type
        self.rebalance = rebalance
        self.debug = debug

        print('loading files index...')
        with open(data_index_fname, 'rb') as f:
            data_dicts = pickle.load(f)


        print('splitting data by dates...')
        dates = np.unique(np.array([d['dt'].date() for d in data_dicts]))
        if isinstance(subsetting_option, float):
            self.curr_subset_dates, _ = train_test_split(dates, test_size=1-subsetting_option)
        elif hasattr(subsetting_option, '__iter__'):
            self.curr_subset_dates = list(set(dates) - set([d for d in subsetting_option]))
        self.dicts = [d for d in data_dicts if d['dt'].date() in self.curr_subset_dates]
        self.targets = np.array([d['observations_TCC'] for d in self.dicts]).astype(np.int32)

        print('balancing the data...')
        self.resample_balanced()


        self.targets_transformer = targets_transformer_np(model_type=self.model_type, batch_size=1)

        self.masks = {}

        self.batch_indices_generator = batches_generator(np.arange(len(self.dicts_resampled)),
                                                         batch_size=self.batch_size,
                                                         shuffle=True)

        self.img_pool = Pool(8)


    @property
    def dates_used(self):
        return self.curr_subset_dates

    def resample_balanced(self):
        if self.rebalance:
            train_sample_size_per_class = np.max([tcc_class.sum() for tcc_class in [self.targets == c for c in np.unique(self.targets[self.targets < 8])]])
            self.dicts_resampled = self.balanced_sample_maker(X=self.dicts, y=self.targets,
                                                              sample_size_per_class=train_sample_size_per_class)
        else:
            self.dicts_resampled = self.dicts


    def balanced_sample_maker(self, X, y, sample_size_per_class, random_seed=None):
        """ return a balanced data set by sampling all classes with sample_size
            current version is developed on assumption that the positive
            class is the minority.

        Parameters:
        ===========
        X: {numpy.ndarrray}
        y: {numpy.ndarray}
        """
        uniq_levels = np.unique(y)
        uniq_counts = {level: sum(y == level) for level in uniq_levels}

        if not random_seed is None:
            np.random.seed(random_seed)

        # find observation index of each class levels
        groupby_levels = {}
        for ii, level in enumerate(uniq_levels):
            obs_idx = [idx for idx, val in enumerate(y) if val == level]
            groupby_levels[level] = obs_idx

        # oversampling on observations of each label
        balanced_copy_idx = []
        for gb_level, gb_idx in groupby_levels.items():
            over_sample_idx = np.random.choice(gb_idx, size=sample_size_per_class, replace=True).tolist()
            balanced_copy_idx += over_sample_idx
        np.random.shuffle(balanced_copy_idx)

        # return [X[i] for i in balanced_copy_idx], [y[i] for i in balanced_copy_idx]
        return [X[i] for i in balanced_copy_idx]


    def preprocess_img(self, fn):
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size[1:], interpolation=cv2.INTER_LANCZOS4)
        return np.expand_dims(img, 0)
        # return img


    def get_mask(self, mask_fname):
        if mask_fname in self.masks.keys():
            return self.masks[mask_fname]
        else:
            msk = self.preprocess_img(mask_fname)
            self.masks[mask_fname] = msk
            return msk

    def __getitem__(self, indices):
        img_fn_batch = [self.dicts_resampled[i]['img_fname_abs'] for i in indices]
        msk_fn_batch = [self.dicts_resampled[i]['mask_fname'] for i in indices]
        trgs = np.array([int(self.dicts_resampled[i]['observations_TCC']) for i in indices])

        imgs = [self.preprocess_img(fn) for fn in img_fn_batch]
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.rollaxis(imgs, -1, 1)
        imgs = t.from_numpy(imgs).float().cuda()

        msks = [self.get_mask(msk_fn) for msk_fn in msk_fn_batch]
        msks = np.concatenate(msks, axis=0)
        msks = np.rollaxis(msks, -1, 1)
        msks = t.from_numpy(msks).float().cuda()

        trgs = self.targets_transformer(trgs)
        trgs = trgs.float()

        if self.debug:
            return (imgs, msks), trgs, img_fn_batch
        else:
            return (imgs, msks), trgs


    def __iter__(self):
        while True:
            indices = next(self.batch_indices_generator)
            yield self.__getitem__(indices)


    def __len__(self):
        return len(self.dicts_resampled)


    def close(self):
        try:
            self.img_pool.close()
        except:
            pass






