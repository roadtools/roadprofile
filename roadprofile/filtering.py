from scipy.signal import butter, lfilter

def _create_highpass_mpd_filter():
    order = 2
    cutoff_frequency = 140 # [mm]
    return butter(order, 1/cutoff_frequency, btype='highpass')

def mpd_highpass(z):
    a, b = _create_highpass_mpd_filter()
    y = lfilter(a, b, z)
    return y
