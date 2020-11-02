try:
    import numpy as np
    from sklearn.model_selection import train_test_split
    import pickle
    import cv2
    import threading
    from .balanced_sampler import balanced_sampler
    from .targets_transformers import *
    import random
except ImportError as e:
    print(e)
    raise ImportError


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    # def next(self):
    #     with self.lock:
    #         return self.it.next()

    def __next__(self):
        with self.lock:
            return next(self.it)



def get_path_i(paths_count):
    """Cyclic generator of paths indices
    """
    current_path_id = 0
    while True:
        yield current_path_id
        current_path_id  = (current_path_id + 1) % paths_count



class InputGenerator:
    def __init__(self,
                 data_index_fname,
                 args,
                 batch_size=16,
                 img_size=(3, 256, 256),
                 # subsetting_option=0.75,
                 model_type='OR',
                 rebalance=True,
                 debug=False):
        print('loading files index...')
        with open(data_index_fname, 'rb') as f:
            data_dicts = pickle.load(f)

        self.data_index_fname = data_index_fname
        self.args = args
        self.img_size = img_size
        self.batch_size = batch_size
        self.model_type = model_type
        self.rebalance = rebalance
        self.debug = debug
        self.index = 0
        self.init_count = 0

        # print('splitting data by dates...')
        # dates = np.unique(np.array([d['dt'].date() for d in data_dicts]))
        # if isinstance(subsetting_option, float):
        #     self.curr_subset_dates, _ = train_test_split(dates, test_size=1 - subsetting_option)
        # elif hasattr(subsetting_option, '__iter__'):
        #     self.curr_subset_dates = list(set(dates) - set([d for d in subsetting_option]))
        # self.dicts = [d for d in data_dicts if d['dt'].date() in self.curr_subset_dates]
        self.dicts = data_dicts
        print('examples number: %d' % len(self.dicts))
        self.targets = np.array([d['TCC'] for d in self.dicts]).astype(np.int32)

        print('balancing the data...')
        if self.rebalance:
            train_sample_size_per_class = np.max([tcc_class.sum() for tcc_class in [self.targets == c for c in np.unique(self.targets[self.targets < 8])]])
            self.dicts = balanced_sampler(X=self.dicts, y=self.targets, sample_size_per_class=train_sample_size_per_class)
            print('rebalanced examples number: %d' % len(self.dicts))


        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.path_id_generator = threadsafe_iter(get_path_i(len(self.dicts)))
        self.imgs = []
        self.trgs = []
        self.msks = []
        self.masks_cached = {}
        if self.args.memcache:
            self.images_cached = {}

        self.targets_transformer = targets_transformer_np(model_type=self.model_type, batch_size=1)

    @property
    def dates_used(self):
        return self.curr_subset_dates

    def get_samples_count(self):
        """ Returns the total number of images needed to train an epoch """
        return len(self.dicts)

    def get_batches_count(self):
        """ Returns the total number of batches needed to train an epoch """
        return int(self.get_samples_count() / self.batch_size)

    def pre_process_input(self, im, lb):
        """ Do your pre-processing here
                    Need to be thread-safe function"""
        return im, lb

    def next(self):
        return self.__iter__()


    def preprocess_img(self, fn):
        img = None
        if self.args.memcache:
            if fn in self.images_cached:
                img = self.images_cached[fn]
        if img is None:
            img = cv2.imread(fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.args.memcache:
                self.images_cached[fn] = img

        return img

    def preprocess_msk(self, fn):
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size[1:], interpolation=cv2.INTER_LANCZOS4)
        return img

    def get_mask(self, mask_fname):
        if mask_fname in self.masks_cached.keys():
            return self.masks_cached[mask_fname]
        else:
            msk = self.preprocess_msk(mask_fname)
            self.masks_cached[mask_fname] = msk
            return msk


    def shuffle(self):
        random.shuffle(self.dicts)


    def __iter__(self):
        while True:
            # In the start of each epoch we shuffle the data paths
            with self.lock:
                if (self.init_count == 0):
                    self.shuffle()
                    self.imgs, self.msks, self.trgs = [], [], []
                    self.init_count = 1
            # Iterates through the input paths in a thread-safe manner
            for path_id in self.path_id_generator:
                img_fn = self.dicts[path_id]['img_fname']
                msk_fn = self.dicts[path_id]['mask_fname']
                trgs = np.array([int(self.dicts[path_id]['TCC'])])

                img = self.preprocess_img(img_fn)
                # img = np.rollaxis(img, -1, 0)

                msk = self.get_mask(msk_fn)
                # msk = np.rollaxis(msk, -1, 0)

                trg = self.targets_transformer(trgs)
                trg = trg.astype(np.float32)

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if (len(self.imgs)) < self.batch_size:
                        self.imgs.append(img)
                        self.msks.append(msk)
                        self.trgs.append(trg)
                    if len(self.imgs) % self.batch_size == 0:
                        imgs_f32 = np.float32(self.imgs)
                        msks_f32 = np.float32(self.msks)
                        trgs_f32 = np.float32(self.trgs)
                        trgs_f32 = np.squeeze(trgs_f32)
                        yield (imgs_f32, msks_f32), trgs_f32
                        self.imgs, self.msks, self.trgs = [], [], []
            # At the end of an epoch we re-init data-structures
            with self.lock:
                random.shuffle(self.dicts)
                self.init_count = 0

    def __call__(self):
        return self.__iter__()
