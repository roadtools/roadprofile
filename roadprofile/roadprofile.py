import numpy as np

_epsilon = np.finfo(np.float32).eps

def interpolate_dropouts(x, y, dropout_criteria):
    drop_outs = _create_dropouts_index(y, dropout_criteria)
    for start, end in _iter_dropout_intervals(drop_outs):
        y[start:end] = np.polyval(
                np.polyfit((x[start-1], x[end + 1]), (y[start-1], y[end+1]), 1),
                x[start:end])

def iter_intervals(x, length):
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

def calculate_mpd(x, y):
    mpd_list = []
    for start, end in iter_intervals(x, 0.1):
        xsub = x[start:end]
        ysub = y[start:end]
        ysub = ysub - np.polyval(np.polyfit(xsub, ysub, 1), xsub)
        mpd = np.mean(_calculate_msd(xsub, ysub))
        mpd_list.append(mpd)
    return np.array(mpd_list)

def envelope_profile_meyer(z, d=0, maxiter=100):
    z = z.copy()
    if d == 0: return z
    C = 0
    i = 1 # Because zero-indexed
    n = len(z)
    dstar = d
    break_count = 0
    while True:
        if i == 1: # Because zero-indexed
            break_count = break_count + 1
        if break_count == maxiter:
            break

        if i - n + 1 < 0:
            d = z[i] - (z[i-1] + z[i+1])/2
            if d - dstar > 0:
                if z[i-1] - z[i] + dstar > 0:
                    z[i+1] = z[i+1] + 2 * (d - dstar)
                    C = C + 1
                else:
                    if z[i+1] - z[i] + dstar < 0:
                        z[i-1] = z[i] - dstar
                        z[i+1] = z[i] - dstar
                        C = C + 1;
                    else:
                        z[i-1] = z[i-1] + 2 * (d - dstar)
                        C = C + 1
            else:
                if d + dstar < 0:
                    z[i] = z[i] - (d-dstar)
                    C = C + 1
            i = i + 1
        else:
            if C == 0:
                break
            else:
                C = 0
                i = 1 # Because zero-indexed
    return z

def _create_dropouts_index(y, dropout_criteria):
    if np.isnan(dropout_criteria):
        drop_outs = np.where(np.isnan(y))[0]
    else:
        drop_outs = np.where(y == dropout_criteria)[0]
    return drop_outs

def _iter_dropout_intervals(drop_outs):
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
            yield drop_outs[idx], drop_outs[idx + count] + 1
            idx += count + 1
            count = 0

def _calculate_msd(xsub, ysub):
    idx = np.where(xsub <= xsub[0] + 0.05)[0][-1] + 1
    msd1 = max(ysub[:idx])
    msd2 = max(ysub[idx:])
    return (msd1, msd2)
