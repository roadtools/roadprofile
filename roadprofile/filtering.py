from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
import numpy as np

from .utils import _epsilon

mpd_butterworth_order = 2
TRUNCATE_THRESHOLD_MM = 5/1000 + _epsilon # 5/1000 = 5 mm

def mpd_butterworth(x, y):
    sampling_rate = np.mean(np.diff(x))
    y = mpd_butterworth_high(y, sampling_rate)
    return mpd_butterworth_low(y, sampling_rate)

def mpd_butterworth_high(z, sampling_rate):
    cutoff_freq = 140/1000 # 140 mm normalized to m
    return butterworth(z, mpd_butterworth_order, cutoff_freq, sampling_rate, 'highpass')

def mpd_butterworth_low(z, sampling_rate):
    cutoff_freq = 3/1000 # 3 mm normalized to m
    return butterworth(z, mpd_butterworth_order, cutoff_freq, sampling_rate, 'lowpass') # TODO find  cutoff-freq in ISO standard.

def butterworth(z, order, cutoff_frequency, sampling_rate, btype): # sampling rate in [samples/mm]
    normalized_frequency = 2 / (cutoff_frequency / sampling_rate) # [half-cycles/sample]
    a, b = butter(order, normalized_frequency, btype=btype)
    return lfilter(a, b, z)

def _create_dropouts_cond(y, criteria):
    if np.isnan(criteria):
        drop_outs = np.isnan(y)
    else:
        drop_outs = y == criteria
    return drop_outs

def _find_invalid_endpoint_intervals(y):
    for n, val in enumerate(y):
        if val: continue
        else: return n

def _handle_startpoints(x, y, cond, truncated):
    end = _find_invalid_endpoint_intervals(cond)
    if x[end - 1] - x[0] > TRUNCATE_THRESHOLD_MM:
        x, y, cond = x[end:], y[end:], cond[end:]
        truncated[0] = end
    else:
        y[:end] = y[end]
        cond[:end] = False
    return x, y, truncated, cond

def _handle_endpoints(x, y, cond, truncated):
    # if there is only one point the left hand side will be 0 thus no truncation.
    # This makes sense since it is required that the sampling interval is smaller than 5 mm
    start = - _find_invalid_endpoint_intervals(reversed(cond))
    if x[-1] - x[start] > TRUNCATE_THRESHOLD_MM:
        x, y, cond = x[:start], y[:start], cond[:start]
        truncated[1] = truncated[1] + start
    else:
        y[start:] = y[start - 1]
        cond[start:] = False
    return x, y, truncated, cond

def interpolate_dropouts(x, y, criteria):
    """
    Replaces all invalid values of `y` with linearly interpolated values based on the neighbouring points (see ISO 13473-1 for more information).

    :param x: Longitudinal distance in meters.
    :param y: Vertical displacement in milimeters.
    :param criteria: Value that defines an invalid measurement in `y`, e.g., `y[n] == criteria` implies that `y[n]` is invalid.
        Thus, `criteria` can be *-9999*, *NaN* or any other special value that indicates an invalid measurement.

    :return: `(y_out, truncated)` where

        * `y_out` Interpolated (and possibly truncated) measurement array.
        * `truncated` A tuple `(start, end)` containing the end-points from the original array, i.e., such that
            `len(y[start:end]) == len(y_out)`. If no truncation have been made the value is `(0, len(y))`.

    *Note* This algorithm does not check if each 100mm segment or the entire profile have enough valid data points.
    """
    y = y.copy()
    if isinstance(criteria, np.ndarray) and criteria.dtype == np.dtype('bool'):
        cond = criteria
    else:
        cond = _create_dropouts_cond(y, criteria)
    truncated = [0, len(x)]
    if cond[0]:
        x, y, truncated, cond = _handle_startpoints(x, y, cond, truncated)
    if cond[-1]:
        x, y, truncated, cond = _handle_endpoints(x, y, cond, truncated)
    negcond = np.logical_not(cond)
    f = interp1d(x[negcond], y[negcond], kind='linear', copy=False)
    y[cond] = f(x[cond])
    return y, tuple(truncated)


def envelope(z, d=0, maxiter=100):
    """
    Calculate the enveloped profile according to [#f1]_ with corrections from [#f2]_.
    Intuitively, the algorithm works by imposing an upper limit on the second derivative of the profile, i.e.,

        z'' <= d

    and adjusting the profile to meet this requirement.

    :param z: Vertical displacement.
    :param d: Empirical parameter associated with the tyre stifness.
    :param maxiter: Maximum number of iterations the algorithm performs. This prevents a potential endless loop from occuring.

    *Note* The current implementation assumes a uniform distance between datapoints.

    .. rubric:: Footnotes
    .. [#f1] von Meier. A., et. al. The influence of texture and sound absorption on the noise of porous road surfaces, PIARC 2nd International Symposium on Road Surface Characteristics, 1992
    .. [#f2] http://www.vegvesen.no/_attachment/58581/binary/2256?fast_title=Dr.+Luc+Goubert%3A+Road+surface+texture+and+traffic+noise

    """
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
