# Need All
import os
import pickle
import numpy as np

from ..tools.tools import check_leftargs, EMPTY

class NumpyAccumulator:
    def __init__(self, logger, input, batch_dim=0, org_type='torch.tensor', **kwargs):
        """
        Parameters
        ----------
        input: [str] Key of value in batch to accumulate
        org_type: [str, 'list', 'torch.tensor', 'np.array'; default='torch.tensor']
            Type of value to accumulate
        batch_dim: [int; default=0] Dimension of batch
        """
        check_leftargs(self, logger, kwargs)
        self.input = input
        self.org_type = org_type
        if org_type in {'tensor', 'torch', 'torch.tensor'}:
            self.converter = lambda x: x.cpu().numpy()
        elif org_type in {'np.array', 'np.ndarray', 'numpy', 'numpy.array', 'numpy.ndarray'}:
            self.converter = EMPTY
        else:
            raise ValueError(f"Unsupported type of config.org_type: {org_type} in NumpyAccumulator")
        self.batch_dim = batch_dim

    def init(self):
        self.accums = []

    def accumulate(self, indices=None):  
        if self.accums[0].ndim == 3:
            mid_values = [arr.shape[1] for arr in self.accums]
            # 最大の真ん中の数を見つける
            max_mid_value = max(mid_values)

        # 各ndarrayを最大の真ん中の数に揃える
            for i in range(len(self.accums)):
                current_mid_value = self.accums[i].shape[1]
                if current_mid_value < max_mid_value:
                    pad_top = (max_mid_value - current_mid_value) // 2
                    pad_bottom = max_mid_value - current_mid_value - pad_top
                    self.accums[i] = np.pad(self.accums[i], ((0, 0), (pad_top, pad_bottom), (0,0)), mode='constant')

        # すべての配列の最大サイズを取得（axis=self.batch_dimの次元を除く）
        max_shape = np.array([a.shape for a in self.accums]).max(axis=0)
        padded_accums = []
        if self.org_type == 'torch.tensor':
            # パディングを適用して、後ろの次元だけ統一
            
            for arr in self.accums:
                pad_width = [(0, 0) for _ in range(arr.ndim)]  # 最初はすべての次元でパディングなし
                pad_width[-1] = (0, max_shape[-1] - arr.shape[-1])  # 最後の次元のみ後ろにパディング
                padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)  # 0でパディング
                padded_accums.append(padded_arr)

        if len(padded_accums) > 0 :
            # パディング後の配列を結合
            accums = np.concatenate(padded_accums, axis=self.batch_dim)
        else:
            accums = np.concatenate(self.accums, axis=self.batch_dim)
        
        if indices is not None:
            accums = accums[indices]
        return accums
        
    def save(self, path_without_ext, indices=None):
        path = path_without_ext + ".npy"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.accumulate(indices=indices))
        
    def __call__(self, batch):
        self.accums.append(self.converter(batch[self.input]))


class ListAccumulator:
    def __init__(self, logger, input, org_type='torch.tensor', batch_dim=None, **kwargs):
        """
        Parameters
        ----------
        input: [str] Key of value in batch to accumulate
        org_type: [str, 'list', 'torch.tensor', 'np.array'] Type of value
                    to accumulate
        """
        check_leftargs(self, logger, kwargs)
        self.input = input
        if org_type == 'list':
            assert batch_dim is None, f"batch_dim cannot be defined when org_type is list"
            self.converter = EMPTY
        else:
            if batch_dim is None: batch_dim = 0
            if org_type in {'tensor', 'torch.tensor'}:
                if batch_dim == 0:
                    self.converter = lambda x: list(x.cpu().numpy())
                else:
                    self.converter = lambda x: list(x.transpose(batch_dim, 0).cpu().numpy())
            elif org_type in {'np.array', 'np.ndarray', 'numpy', 'numpy.array', 'numpy.ndarray'}:
                if batch_dim == 0:
                    self.converter = lambda x: list(x)
                else:
                    self.converter = lambda x: list(x.swapaxes(0, batch_dim))
    def init(self):
        self.accums = []
    def accumulate(self, indices=None):
        if indices is not None:
            accums = np.array(self.accums, dtype=object)
            return accums[indices].tolist()
        else:
            return self.accums
    def save(self, path_without_ext, indices=None):
        path = path_without_ext + '.pkl'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path_without_ext}.pkl", 'wb') as f:
            pickle.dump(self.accumulate(indices=indices), f)
    def __call__(self, batch):
        self.accums += self.converter(batch[self.input])

accumulator_type2class = {
    'numpy': NumpyAccumulator,
    'list': ListAccumulator
}

def get_accumulator(type, **kwargs):
    return accumulator_type2class[type](**kwargs)