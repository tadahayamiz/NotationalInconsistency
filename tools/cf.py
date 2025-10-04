import concurrent.futures as cf

def _chunk_process(process, chunk_array, verbose):
    if verbose:
        print("*", end='', flush=True)
    return [process(point) for point in chunk_array]

def cf_apply(process, array, max_workers, verbose=True):

    with cf.ProcessPoolExecutor(max_workers=max_workers) as e:
        futures = []
        for i_worker in range(max_workers):
            chunk_size = len(array) // (max_workers - i_worker)
            futures.append(e.submit(_chunk_process, process, array[:chunk_size], verbose))
            array = array[chunk_size:]
        result = []
        for future in futures:
            result += future.result()
        return result
