import numpy as np

_epsilon = np.finfo(np.float32).eps

def apply_each_evaluation_length_and_save_result(x, y, func, nmean, seglen):
    result_list = []
    x_list = []

    check = nmean - 1
    value_array = np.zeros((nmean,))
    x_list.append(x[0])
    for n, (start, end) in enumerate(iter_intervals(x, seglen)):
        xsub, ysub = x[start:end], y[start:end]
        idx = n % nmean
        value_array[idx] = func(xsub, ysub)
        if idx == check:
            x_list.append(xsub[-1])
            result_list.append(np.mean(value_array))
    return x_list, result_list

def iter_intervals(x, length=0.1):
    length = length - _epsilon # dirty fix because, e.g., 0.21 - 0.11 >= 0.1 is False
    last_idx = 0
    max_idx = len(x)
    while last_idx < len(x) and (x[-1] - x[last_idx]) >= length:
        indices = np.where(x[last_idx:max_idx] >= x[last_idx] + length)[0]
        next_idx = min(indices) + 1
        yield last_idx, last_idx + next_idx
        max_idx = last_idx + (next_idx * 3)
        last_idx += next_idx
        if len(x) <= max_idx:
            max_idx = len(x)

def _iter_intervals_of_true(drop_outs):
    idx = 0
    count = 0
    while idx < len(drop_outs):
        try:
            consecutive = (drop_outs[idx + 1] - drop_outs[idx]) == 1
        except IndexError:
            yield drop_outs[idx - count], drop_outs[idx] + 1
            break
        if consecutive:
            count += 1
            idx += 1
        else:
            yield drop_outs[idx - count], drop_outs[idx] + 1
            idx += 1
            count = 0
