from operator import xor
import pickle
import numpy as np
import torch
# Buckets: moved to past_activities/buckets

class Buckets2:
    def __init__(self, input_tokens_list, target_tokens_list, padding_value=0,
        features=None,
        min_bucket_len=None, sup_bucket_len=None,
        num_buckets=None, bucket_bins=None, right=None, add_lower_margin=True, add_upper_margin=True,
        batch_size=None, batch_sizes=None, num_tokens=None, seed=0,
        n_epoch=float('inf'), 
        return_index=False, device='cpu', logger=None, **kwargs):
        """
        Parameters
        ----------
        input_tokens_list: array_like of array_like of int
        target_tokens_list: array_like of array_like of int
        features: None, np.array [n_sample], np.array[n_sample, n_feature],
            Features to iterate. If None, iteration is omitted.
        min_bucket_len: None or int
            If None, shortest length in input_tokens_list is substituted.
        sup_bucket_len: None or int
            If None, longest length in target_tokens_list + 1 is substituted.
        """
        # ckeck args
        if int(batch_size is not None)+int(batch_sizes is not None)+int(num_tokens is not None) != 1:
            raise ValueError(f"Specify exactly one of batch_size({batch_size}), batch_sizes({batch_sizes}), num_tokens({num_tokens})")
        if logger is not None and len(kwargs) > 0:
            logger.debug(f"args not used: {kwargs.keys()}")
        
        # process token list
        self.input_tokens_list = input_tokens_list
        self.target_tokens_list = target_tokens_list
        self.features = features
        self.input_lens = np.array([len(input_) for input_ in input_tokens_list])
        self.target_lens = np.array([len(target) for target in target_tokens_list])


        # bucketing
        if bucket_bins is not None:
            assert (min_bucket_len is None) and (sup_bucket_len is None) \
                and (num_buckets is None), f"Specify either bucket_bins({bucket_bins})"+\
                f" or min_bucket_len({min_bucket_len}), sup_bucket_len({sup_bucket_len}), num_buckets({num_buckets})"
            assert right is not None, "Specify right."
            num_buckets = len(bucket_bins)-1
        else:
            assert num_buckets is not None, "Specify num_buckets."
            assert right is None
            right = False
            if min_bucket_len is None:
                min_bucket_len = self.input_lens.min()
            if sup_bucket_len is None:
                sup_bucket_len = self.input_lens.max()+1
            bucket_bins = np.linspace(min_bucket_len, sup_bucket_len, num_buckets+1)
        bucket_ids = np.digitize(self.input_lens, bucket_bins, right=right) - 1
        self.num_buckets = num_buckets
        if add_lower_margin:
            self.num_buckets += 1
            bucket_ids += 1
            bucket_ids[bucket_ids < 0] = 0
        if add_upper_margin:
            self.num_buckets += 1
            bucket_ids[bucket_ids >= self.num_buckets] = self.num_buckets-1
        self.bucket_ids = bucket_ids
        self.padding_value = padding_value
        self.features = features


        # batch
        if batch_size is not None:
            self.batch_sizes = [batch_size]*self.num_buckets
        elif batch_sizes is not None:
            self.batch_sizes = batch_sizes
        elif num_tokens is not None:
            self.batch_sizes = []
            for i_bucket in range(self.num_buckets):
                bucket_input_lens = self.input_lens[self.bucket_ids == i_bucket]
                if len(bucket_input_lens) > 0:
                    batch_size = num_tokens // np.max(bucket_input_lens)
                else:
                    batch_size = 1
                self.batch_sizes.append(batch_size)
        print(f"bucket bins: {bucket_bins}")
        print(f"batch sizes: {self.batch_sizes}")
        assert len(self.batch_sizes) == self.num_buckets, \
            f"len(batch_sizes)({len(self.batch_sizes)}) != self.num_buckets({self.num_buckets})"

        self.rstate = np.random.RandomState(seed=seed)
        self.n_epoch = n_epoch
        self.device = device
        self.return_index = return_index
        self.init_state()

    def init_state(self):
        self.epoch = -1
        self.end_epoch = 0
        self.n_batch = 0
        for bucket_id, batch_size in enumerate(self.batch_sizes):
            bucket_idxs = np.where(self.bucket_ids == bucket_id)[0]
            self.n_batch += int(np.ceil(len(bucket_idxs)/batch_size))
        self.i_batch_idx = -1
        self.step = 0

    def __iter__(self):
        return self
    def __next__(self):
        self.i_batch_idx  = (self.i_batch_idx+1) % self.n_batch
        self.step += 1
        if self.i_batch_idx == self.n_batch - 1:
            self.end_epoch += 1
        if self.i_batch_idx == 0:
            self.epoch += 1
            if self.epoch == self.n_epoch:
                self.init_state()
                raise StopIteration()
            self.batch_idxs = []
            for bucket_id, batch_size in zip(range(self.num_buckets), self.batch_sizes):
                bucket_idxs = np.where(self.bucket_ids == bucket_id)[0]
                self.rstate.shuffle(bucket_idxs)
                for batch_start in range(0, len(bucket_idxs), batch_size):
                    self.batch_idxs.append(bucket_idxs[batch_start:batch_start+batch_size])
            self.rstate.shuffle(self.batch_idxs)
            assert self.n_batch == len(self.batch_idxs)
        batch_idx = self.batch_idxs[self.i_batch_idx]
        input_len_batch = self.input_lens[batch_idx]
        target_len_batch = self.target_lens[batch_idx]
        input_max_len = np.max(self.input_lens[batch_idx])
        output_max_len = np.max(self.target_lens[batch_idx])
        input_batch = np.full((len(batch_idx), input_max_len),
            fill_value=self.padding_value)
        target_batch = np.full((len(batch_idx), output_max_len),
            fill_value=self.padding_value)
        for i_in_batch, idx in enumerate(batch_idx):
            input_batch[i_in_batch, :self.input_lens[idx]] = \
                np.array(self.input_tokens_list[idx])
            target_batch[i_in_batch, :self.target_lens[idx]] = \
                np.array(self.target_tokens_list[idx])
        batch = (torch.tensor(input_batch, dtype=torch.long, device=self.device),
                torch.tensor(target_batch, dtype=torch.long, device=self.device),
                torch.tensor(input_len_batch, dtype=torch.long, device=self.device),
                torch.tensor(target_len_batch, dtype=torch.long, device=self.device),)
        if self.features is not None:
            feature_batch = self.features[batch_idx]
            batch = batch + (torch.tensor(feature_batch, dtype=torch.float, device=self.device), )
        if self.return_index:
            batch = (torch.tensor(batch_idx, dtype=torch.long, device=self.device),) + batch
        return batch
    def save_iter(self, file):
        input_tokens_list = self.input_tokens_list
        target_tokens_list = self.target_tokens_list
        input_lens = self.input_lens
        target_lens = self.target_lens
        del self.input_tokens_list, self.target_tokens_list, self.input_lens, self.target_lens
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        self.input_tokens_list = input_tokens_list
        self.target_tokens_list = target_tokens_list
        self.input_lens = input_lens
        self.target_lens = target_lens

    @classmethod
    def load_iter(cls, file, input_tokens_list, target_tokens_list):
        with open(file, 'rb') as f:
            answer = pickle.load(f)
        answer.input_tokens_list = input_tokens_list
        answer.target_tokens_list = target_tokens_list
        answer.input_lens = np.array([len(input_) for input_ in input_tokens_list])
        answer.target_lens = np.array([len(target) for target in target_tokens_list])
        return answer