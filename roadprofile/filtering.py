from scipy.signal import butter, lfilter
from numpy import mean, diff

mpd_butterworth_order = 2

def mpd_butterworth(x, y):
    sampling_rate = mean(diff(x))
    y = mpd_butterworth_high(y, sampling_rate)
    return mpd_butterworth_low(y, sampling_rate)

def mpd_butterworth_high(z, sampling_rate):
    cutoff_freq = 140
    return butterworth(z, mpd_butterworth_order, cutoff_freq, sampling_rate, 'highpass')

def mpd_butterworth_low(z, sampling_rate):
    cutoff_freq = 3
    return butterworth(z, mpd_butterworth_order, cutoff_freq, sampling_rate, 'lowpass') # TODO find  cutoff-freq in ISO standard.

def butterworth(z, order, cutoff_frequency, sampling_rate, btype): # sampling rate in [samples/mm]
    normalized_frequency = 2 / (cutoff_frequency / sampling_rate) # [half-cycles/sample]
    a, b = butter(order, normalized_frequency, btype=btype)
    return lfilter(a, b, z)

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
