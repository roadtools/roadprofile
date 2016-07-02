import numpy as np
from numpy import mean

from .utils import apply_each_evaluation_length_and_save_result
from .filtering import mpd_butterworth

def calculate_mpd(x, y, method='slope', seglen=0.1, nmean=10):
    """
    Calculate mean profile depth (MPD) according to ISO-13473-1.

    :param x: Longitudinal distance in meters.
    :param y: Vertical displacement in milimeters.
    :param str method: Profile filtering method used. '*slope*' uses slope suppression according to ISO-13473-1, '*butterworth*' uses the high- and low-pass filtering according to ISO-13473-1, and '*no filtering*' applies no filtering at all.
    :param int seglen: Length of evaluation segment used in MPD calculations. Default is 10 cm as specified in ISO-13473-1. Each segment is chosen to be the least segment that is equal to or larger than `seglen`.
    :param int nmean: Number of Mean Segment Depth (MSD) values that is being averaged into one MPD value. Default is 10 which is the *least* value recommended in the ISO standard (with seglen=0.1).
    :return: `(x_interval, mpd)` where

        * `x_interval` is an array with length `len(mpd) + 1` of start/end values of the consecutive intervals where MPD have been calculated.
        * `mpd` array of calculated MPD values.

    """

    if method=='slope':
        x_list, mpd_list = apply_each_evaluation_length_and_save_result(x,y, _calc_msd_w_slopesupress, nmean, seglen)
    elif method=='butterworth':
        y = mpd_butterworth(x, y)
        x_list, mpd_list = apply_each_evaluation_length_and_save_result(x,y, _calculate_msd, nmean, seglen)
    elif method=='no filtering':
        x_list, mpd_list = apply_each_evaluation_length_and_save_result(x,y, _calculate_msd, nmean, seglen)
    else:
        raise Exception('method "{}" not known.'.format(method))
    return np.array(x_list), np.array(mpd_list)

def _calc_msd_w_slopesupress(xsub, ysub):
    ysub = ysub - np.polyval(np.polyfit(xsub, ysub, 1), xsub)
    return _calculate_msd(xsub, ysub)

def _calculate_msd(xsub, ysub):
    idx = np.where(xsub <= xsub[0] + 0.05)[0][-1] + 1
    msd1 = max(ysub[:idx])
    msd2 = max(ysub[idx:])
    return mean(msd1, msd2)
