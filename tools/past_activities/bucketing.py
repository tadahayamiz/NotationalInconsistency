import numpy as np
import torch

class Buckets:
    def __init__(self, input_tokens_list, target_tokens_list, features=None,
        min_bucket_len=None, sup_bucket_len=None,
        num_buckets=10, padding_value=0, **kwargs):
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
        self.input_tokens_list = input_tokens_list
        self.target_tokens_list = target_tokens_list
        self.features = features
        self.input_lens = np.array([len(input_) for input_ in input_tokens_list])
        self.target_lens = np.array([len(target) for target in target_tokens_list])
        if min_bucket_len is None:
            min_bucket_len = self.input_lens.min()
        if sup_bucket_len is None:
            sup_bucket_len = self.input_lens.max()+1
        cast_value = (sup_bucket_len - min_bucket_len) / num_buckets
        minimum = min_bucket_len / cast_value
        bucket_ids = (self.input_lens / cast_value - minimum + 1).astype(int)
        self.num_buckets = num_buckets+2
        bucket_ids[bucket_ids < 0] = 0
        bucket_ids[bucket_ids >= self.num_buckets] = self.num_buckets-1
        self.bucket_max_lens = np.append(np.linspace(min_bucket_len, sup_bucket_len, num_buckets+1)-1,
            [self.input_lens.max()]).astype(int)
        self.bucket_ids = bucket_ids
        self.padding_value = padding_value

    def iterate(self, batch_size=None, num_tokens=None, seed=0, n_epoch=1,
            add_lower_margin=True, add_upper_margin=True, device='cpu', return_index=False, **kwargs):
        if (batch_size is None) == (num_tokens is None):
            raise ValueError("Specify either batch_size XOR num_tokens.")
        if batch_size is None:
            batch_sizes = [int(num_tokens/max_len) for max_len in self.bucket_max_lens]
        elif type(batch_size) == int:
            batch_sizes = [batch_size]*self.num_buckets
        else:
            batch_sizes = batch_size
        rstate = np.random.RandomState(seed=seed)
        self.epoch = 0
        while self.epoch < n_epoch:
            batch_idxs = []
            for bucket_id, batch_size in zip(range(self.num_buckets), batch_sizes):
                if ((bucket_id == 0) and (not add_lower_margin)) or \
                    ((bucket_id == self.num_buckets-1) and (not add_upper_margin)):
                    continue
                bucket_idxs = np.where(self.bucket_ids == bucket_id)[0]
                rstate.shuffle(bucket_idxs)
                for batch_start in range(0, len(bucket_idxs), batch_size):
                    batch_idxs.append(bucket_idxs[batch_start:batch_start+batch_size])
            batch_idxs = np.array(batch_idxs, dtype=object)
            rstate.shuffle(batch_idxs)
            for batch_idx in batch_idxs:
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
                batch = (torch.tensor(input_batch, dtype=torch.long, device=device),
                        torch.tensor(target_batch, dtype=torch.long, device=device),
                        torch.tensor(input_len_batch, dtype=torch.long, device=device),
                        torch.tensor(target_len_batch, dtype=torch.long, device=device),)
                if self.features is not None:
                    feature_batch = self.features[batch_idx]
                    batch = batch + (torch.tensor(feature_batch, dtype=torch.float, device=device), )
                if return_index:
                    batch = (torch.tensor(batch_idx, dtype=torch.long, device=device),) + batch
                yield batch
                
            self.epoch += 1