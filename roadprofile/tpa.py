import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from .utils import apply_each_evaluation_length_and_save_result

def calculate_tpa(x, y, nmean=10, seglen=0.1):
    x_list, tpa_list = apply_each_evaluation_length_and_save_result(x,y, _calc_tpa_core, nmean, seglen)
    return np.array(x_list), np.array(tpa_list)

def _calc_tpa_core(xsub, ysub, threshold): # threshold refers to fraction of lowest data should be discarded
    ysub += abs(min(ysub))
    f = interp1d(xsub, ysub, kind='linear')
    size = len(xsub) * 20
    x_interp = np.linspace(xsub[0], xsub[-1], size)
    y_interp = f(x_interp)
    y_sorted = np.sort(y_interp)
    idx = round(size * threshold)
    cutoff_length = y_sorted[idx]
    y_interp -= cutoff_length
    y_interp[np.where(y_interp < 0)[0]] = 0
    return cumtrapz(x_interp, y_interp)
