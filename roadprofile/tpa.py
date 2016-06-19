import numpy as np
from scipy.integrate import cumtrapz

from .utils import apply_each_evaluation_length_and_save_result

def calculate_tpa(x, y, nmean=10, seglen=0.1):
    def calc_tpa_core(xsub, ysub):
        ysub += abs(min(ysub))
        return cumtrapz(xsub, ysub)

    x_list, tpa_list = apply_each_evaluation_length_and_save_result(x,y, calc_tpa_core, nmean, seglen)
    return np.array(x_list), np.array(tpa_list)
