import numpy as np

from .utils import apply_each_evaluation_length_and_save_result

def calculate_mpd(x, y, nmean=10, seglen=0.1):
    def calc_mpd_core(xsub, ysub):
        ysub = ysub - np.polyval(np.polyfit(xsub, ysub, 1), xsub)
        return np.mean(_calculate_msd(xsub, ysub))

    x_list, mpd_list = apply_each_evaluation_length_and_save_result(x,y, calc_mpd_core, nmean, seglen)
    return np.array(x_list), np.array(mpd_list)

def _calculate_msd(xsub, ysub):
    idx = np.where(xsub <= xsub[0] + 0.05)[0][-1] + 1
    msd1 = max(ysub[:idx])
    msd2 = max(ysub[idx:])
    return (msd1, msd2)
