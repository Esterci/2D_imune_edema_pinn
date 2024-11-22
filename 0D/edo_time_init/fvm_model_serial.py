import numpy as np


def fb(Cb, Cn, cb, lambd_nb):
    return (cb - lambd_nb * Cn) * Cb


def fn(Cb, Cn, y_n, Cn_max, lambd_bn, mi_n):
    return y_n * Cb * (Cn_max - Cn) - lambd_bn * Cn * Cb - mi_n * Cn


def solve(
    Cb_final,
    Cn_final,
    initial_cond,
    n_it,
    cb,
    lambd_nb,
    y_n,
    Cn_max,
    lambd_bn,
    mi_n,
    k,
    phi,
):

    for i in range(len(initial_cond)):

        Cb_final[i][0] = initial_cond[i]
        Cn_final[i][0] = 0

        for time in range(1, n_it):

            Cb_final[i][time] = (k / phi) * (
                fb(Cb_final[i][time - 1], Cn_final[i][time - 1], cb, lambd_nb)
            ) + Cb_final[i][time - 1]

            Cn_final[i][time] = (k / phi) * (
                fn(
                    Cb_final[i][time - 1],
                    Cn_final[i][time - 1],
                    y_n,
                    Cn_max,
                    lambd_bn,
                    mi_n,
                )
            ) + Cn_final[i][time - 1]

    return Cb_final, Cn_final
