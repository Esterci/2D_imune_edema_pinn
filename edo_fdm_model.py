import numpy as np
from utils import ProgBar


def fb(Cb, Cn, cb, lambd_nb):
    return (cb - lambd_nb * Cn) * Cb


def fn(Cb, Cn, Cn_max, y_n, lambd_bn, mi_n):
    return y_n * Cb * (Cn_max - Cn) - lambd_bn * Cn * Cb - mi_n * Cn


def solve_pde(
    k,
    phi,
    cb,
    lambd_nb,
    mi_n,
    lambd_bn,
    y_n,
    t,
    Cn_max,
    size_tt,
):

    Cn_final = np.zeros((size_tt))
    Cb_final = np.zeros((size_tt))

    Cn_new = 0
    Cb_new = 0.2

    # bar = ProgBar(size_tt, "Calculando ...")

    for time in range(size_tt):

        Cn_old = Cn_new
        Cb_old = Cb_new

        Cb_new = (k / phi) * (fb(Cb_old, Cn_old, cb, lambd_nb)) + Cb_old

        Cn_new = (k / phi) * (fn(Cb_old, Cn_old, Cn_max, y_n, lambd_bn, mi_n)) + Cn_old

        Cb_final[time] = Cb_new
        Cn_final[time] = Cn_new

        # bar.update()

    return Cb_final, Cn_final
