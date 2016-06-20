from scipy.signal import butter, lfilter
from numpy import mean, diff

def mpd_butterworth(x, y):
    sampling_rate = mean(diff(x))
    y = mpd_butterworth_high(y, sampling_rate)
    return mpd_butterworth_low(y, sampling_rate)

def mpd_butterworth_high(z, sampling_rate):
    return butterworth(z, 2, 140, sampling_rate, 'highpass')

def mpd_butterworth_low(z, sampling_rate):
    return butterworth(z, 2, 3, sampling_rate, 'lowpass') # TODO find  cutoff-freq in ISO standard.

def butterworth(z, order, cutoff_frequency, sampling_rate, btype): # sampling rate in [samples/mm]
    normalized_frequency = 2 / (cutoff_frequency / sampling_rate) # [half-cycles/sample]
    a, b = butter(order, normalized_frequency, btype='highpass')
    return lfilter(a, b, z)

def envelope(z, d=0, maxiter=100):
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
