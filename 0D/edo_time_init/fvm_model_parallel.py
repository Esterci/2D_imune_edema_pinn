from numba import cuda

@cuda.jit(device=True)
def cu_fb(Cb, Cn, cb, lambd_nb):
    return (cb - lambd_nb * Cn) * Cb


@cuda.jit(device=True)
def cu_fn(Cb, Cn, y_n, Cn_max, lambd_bn, mi_n):
    return y_n * Cb * (Cn_max - Cn) - lambd_bn * Cn * Cb - mi_n * Cn


@cuda.jit()
def cu_solve_pde(
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
    phi
):

    # Domain position
    i = cuda.grid(1)

    Cb_final[i][0] = initial_cond[i]

    for time in range(1, n_it):

        Cb_final[i][time] = (k / phi) * (
            cu_fb(Cb_final[i][time - 1], Cn_final[i][time - 1], cb, lambd_nb)
        ) + Cb_final[i][time - 1]

        Cn_final[i][time] = (k / phi) * (
            cu_fn(
                Cb_final[i][time - 1],
                Cn_final[i][time - 1],
                y_n,
                Cn_max,
                lambd_bn,
                mi_n,
            )
        ) + Cn_final[i][time - 1]