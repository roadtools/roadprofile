import numpy as np
from numpy import mean

from .utils import apply_each_evaluation_length_and_save_result
from .filter import mpd_butterworth

def calculate_mpd(x, y, method='slope', nmean=10, seglen=0.1):
    if method=='slope':
        x_list, mpd_list = apply_each_evaluation_length_and_save_result(x,y, _calc_msd_w_slopesupress, nmean, seglen)
    elif method=='butterworth':
        y = mpd_butterworth(x, y)
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
