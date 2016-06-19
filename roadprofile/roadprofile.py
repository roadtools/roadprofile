import numpy as np

from utils import apply_each_evaluation_length_and_save_result

def calculate_mpd(x, y, nmean=10, seglen=0.1):
    def calc_mpd_core(xsub, ysub):
        ysub = ysub - np.polyval(np.polyfit(xsub, ysub, 1), xsub)
        return np.mean(_calculate_msd(xsub, ysub))

    x_list, mpd_list = apply_each_evaluation_length_and_save_result(x,y, calc_mpd_core, nmean, seglen)
    return np.array(x_list), np.array(mpd_list)

def envelope_profile(z, d=0, maxiter=100):
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

def _calculate_msd(xsub, ysub):
    idx = np.where(xsub <= xsub[0] + 0.05)[0][-1] + 1
    msd1 = max(ysub[:idx])
    msd2 = max(ysub[idx:])
    return (msd1, msd2)
