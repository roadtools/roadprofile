import numpy as np
from scipy.interpolate import interp1d

from .utils import apply_each_evaluation_length_and_save_result

def calculate_tpa(x, y, nmean=10, seglen=0.1):
    """
    Calculate the Texture Penetration Area (TPA) as described in section 4.3 of [#f3]_.


    :param x: Longitudinal distance in meters.
    :param y: Vertical displacement in milimeters.
    :param int seglen: Same as :func:`.calculate_mpd`.
    :param int nmean: Same as :func:`.calculate_mpd`.
    :return: `(x_interval, tpa)` where

        * `x_interval` is an array with length `len(mpd) + 1` of start/end values of the consecutive intervals where TPA have been calculated.
        * `tpa` array of calculated TPA values.

    .. rubric:: Footnotes
    .. [#f3] http://forskning.ruc.dk/site/en/publications/id%287ea66167-850c-49ad-95a1-67bd4d8c4957.html
    """
    x_list, tpa_list = apply_each_evaluation_length_and_save_result(x, y, _calc_tpa_core, nmean, seglen)
    return np.array(x_list), np.array(tpa_list)

def _calc_tpa_core(xsub, ysub, threshold): # threshold refers to fraction of lowest data should be discarded
    ysub += abs(min(ysub))
    f = interp1d(xsub, ysub, kind='linear')
    size = len(xsub) * 50 # how many interpolation points per data sample
    x_interp = np.linspace(xsub[0], xsub[-1], size)
    y_interp = f(x_interp)
    y_sorted = np.sort(y_interp)
    idx = size - round(size * threshold)
    cutoff_length = y_sorted[idx]
    y_interp -= cutoff_length
    y_interp[np.where(y_interp < 0)[0]] = 0
    return np.trapz(y_interp, x_interp)
