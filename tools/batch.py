import numpy as np
import copy
"""
def make_batch_masks(data_index, batch_size=None, n_batch=None, random_state=0,
                     residue='ignore'):
    Make list of batch_masks which splits train data into batches

    Parameters
    ----------
    data_index: array_like or set of int
        Indices of data to split into batches.
    batch_size: int or None
        Number of data in each batch
    n_batch: int or None
        Number of batch
        It is invalid when batch_size is specified.
    residue: str
        How to treat residues of batches (i.e. n_data%batch_size or n_data%n_batch)
        'ignore': Do not assign any batch to residues
        'new': residues consist new batch
        'include': residues are classified to existing batchess
    random_state: int
        Seed of RNG which classifies batches

    Returns
    -------
    batch_masks: list of np.ndarray(batch_size) of int
        list of indices of data included in each batch.
        batch_size may vary among batch_masks due to residues.

    Notes
    -----
    Appoint at least either batch_size or n_batch.
    n_data = len(data_index)
    rstate = np.random.RandomState(random_state)
    if batch_size is None:
        batch_size = int(n_data // n_batch)
    else:
        if n_data <= batch_size:
            return [data_index]
    n_batch = int(n_data // batch_size)
    if n_batch*batch_size < n_data and residue == 'include':
        batch_size = n_data // n_batch

    shuffled_index = np.array(data_index)
    rstate.shuffle(shuffled_index)
    batch_masks = []
    for i_batch in range(n_batch):
        batch_masks.append(
            shuffled_index[i_batch*batch_size:(i_batch+1)*batch_size])

    if batch_size*n_batch < n_data:
        if residue == 'ignore':
            pass
        elif residue == 'new':
            batch_masks.append(shuffled_index[batch_size*n_batch:])
        elif residue == 'include':
            for i, residue_index in enumerate(shuffled_index[batch_size*n_batch:]):
                batch_masks[i] = np.append(batch_masks[i], residue_index)

    return batch_masks
"""
import numpy as np

#旧make_batch_masks2
def make_batch_masks(data_index, batch_size=None, n_batch=None, random_state=0,
                     residue='ignore', shuffle=True):
    """
    Make list of batch_masks which splits train data into batches

    Parameters
    ----------
    data_index: array_like or set of int
        Indices of data to split into batches.
    batch_size: int or None
        Number of data in each batch
    n_batch: int or None
        Number of batch
        It is invalid when batch_size is specified.
    residue: str
        How to treat residues of batches (i.e. n_data%batch_size or n_data%n_batch)
        'ignore': Do not assign any batch to residues
        'new': residues consist new batch
        'include': residues are classified to existing batchess
    shuffle: bool
        If True, datas are batched randomly.
        if False, datas are batched in order of data_index
    random_state: int
        Seed of RNG which classifies batches

    Returns
    -------
    batch_masks: list of np.ndarray(batch_size) of int
        list of indices of data included in each batch.
        batch_size may vary among batch_masks due to residues.

    Notes
    -----
    Appoint at least either batch_size or n_batch.
    """
    data_index = copy.deepcopy(data_index)
    assert (batch_size is not None) or (n_batch is not None)
    n_data = len(data_index)
    rstate = np.random.RandomState(random_state)
    if residue == 'ignore':
        if batch_size is not None:
            n_batch = n_data // batch_size
        n_datas_batch = np.full(n_batch, fill_value=batch_size)
    elif residue == 'include':
        if 0 < n_data < batch_size:
            n_datas_batch = [n_data]
        else:
            if batch_size is not None:
                n_batch = n_data // batch_size
            n_datas_batch = []
            for i in range(n_batch):
                n_datas_batch.append((n_data-sum(n_datas_batch)) // (n_batch-i))
    elif residue == 'new':
        if batch_size is not None:
            n_batch = n_data // batch_size
        n_datas_batch = list(np.full(n_batch, fill_value=batch_size))
        res = n_data - n_batch*batch_size 
        if res > 0:
            n_datas_batch.append(res)
    else:
        raise ValueError(f"Unsupported type of handling residue: {residue}")

    data_index = np.array(data_index)
    if shuffle:
        rstate.shuffle(data_index)
    n_batch = len(n_datas_batch)
    batch_masks = []
    for n_data_batch in n_datas_batch:
        batch_masks.append(data_index[:n_data_batch])
        data_index = data_index[n_data_batch:]
    return batch_masks

