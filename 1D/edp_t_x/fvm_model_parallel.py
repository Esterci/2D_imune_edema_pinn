from numba import cuda


# Função que descreve a taxa de variação da concentração de bactérias (Cb)
@cuda.jit(device=True, fastmath=True)
def cu_fb(Cb, Cn, i, j, cb, lambd_nb):
    # O crescimento de bactérias é reduzido pela presença de neutrófilos (Cn)
    # por um fator lambd_nb
    return (cb - lambd_nb * Cn[i, j]) * Cb[i, j]


# Função que descreve a taxa de variação da concentração de neutrófilos (Cn)
@cuda.jit(device=True, fastmath=True)
def cu_fn(Cb, Cn, source_points, i, j, y_n, Cn_max, lambd_bn, mi_n):
    # Crescimento dos neutrófilos depende da presença de bactérias (Cb)
    # Também considera uma taxa de decaimento natural (mi_n) e a interação com
    # bactérias (lambd_bn)
    return (
        y_n * Cb[i, j] * (Cn_max - Cn[i, j]) * source_points[i, j]
        - lambd_bn * Cn[i, j] * Cb[i, j]
        - mi_n * Cn[i, j]
    )


# Função para aplicar condições iniciais à concentração de bactérias (Cb)
@cuda.jit(device=True, fastmath=True)
def cu_apply_initial_conditions(ini_cond, Cb, cx, cy, radius, size_x, size_y):
    for i in range(size_x):
        for j in range(size_y):
            # Calculate distance from center to each point
            if (i - cx) ** 2 + (j - cy) ** 2 <= radius**2:
                Cb[i][j] = ini_cond  # Set point inside the circle to 1

    return Cb


# Função principal para resolver as equações diferenciais parciais usando diferenças finitas
@cuda.jit(fastmath=True)
def cu_solve_pde(
    Cb_buf_0,
    Cn_buf_0,
    Cb_buf_1,
    Cn_buf_1,
    Cb_final,
    Cn_final,
    leu_source_points,
    size_t,
    size_x,
    size_y,
    h,
    k,
    Db,
    Dn,
    phi,
    cb,
    lambd_nb,
    mi_n,
    lambd_bn,
    y_n,
    Cn_max,
    X_nb,
    initial_cond,
    center,
    radius,
):

    # Domain position
    i, j = cuda.grid(2)

    # Aplicando condições iniciais para a concentração de bactérias
    cx_real, cy_real = center
    cx_disc = cx_real / h
    cy_disc = cy_real / h
    radius_disc = radius / h

    # Aplicando condições iniciais para a concentração de bactérias
    Cb_buf_0 = cu_apply_initial_conditions(
        initial_cond, Cb_buf_0, cx_disc, cy_disc, radius_disc, size_x, size_y
    )

    # Armazenando as condições iniciais
    Cb_final[0][i, j] = Cb_buf_0[i, j]
    Cn_final[0][i, j] = Cn_buf_0[i, j]

    # Don't continue if our index is outside the domain

    if i >= size_x or j >= size_y:
        return

    # Prepare to do a grid-wide synchronization later

    grid = cuda.cg.this_grid()

    # Loop sobre o tempo
    for time in range(1, size_t):
        # Atualizando as concentrações anteriores (passo temporal anterior)

        if (time % 2) == 0:

            Cn_old = Cn_buf_1
            Cb_old = Cb_buf_1

            Cn_new = Cn_buf_0
            Cb_new = Cb_buf_0

        else:
            Cn_old = Cn_buf_0
            Cb_old = Cb_buf_0

            Cn_new = Cn_buf_1
            Cb_new = Cb_buf_1

        diff_Cb_right = 0 if (i == size_x - 1) else (Cb_old[i + 1, j] - Cb_old[i, j])

        diff_Cb_left = 0 if (i == 0) else (Cb_old[i, j] - Cb_old[i - 1, j])

        diff_Cb_up = 0 if (j == size_y - 1) else (Cb_old[i, j + 1] - Cb_old[i, j])

        diff_Cb_down = 0 if (j == 0) else (Cb_old[i, j] - Cb_old[i, j - 1])

        # Atualizando as concentrações de bactérias
        Cb_new[i][j] = (
            (k * Db)
            / (h * h * phi)
            * (diff_Cb_right - diff_Cb_left + diff_Cb_up - diff_Cb_down)
            + (k / phi) * cu_fb(Cb_old, Cn_old, i, j, cb, lambd_nb)
            + Cb_old[i, j]
        )

        diff_Cn_right = 0 if i == size_x - 1 else ((Cn_old[i + 1, j] - Cn_old[i, j]))

        diff_Cn_left = 0 if i == 0 else ((Cn_old[i, j] - Cn_old[i - 1, j]))

        diff_Cn_up = 0 if j == size_y - 1 else ((Cn_old[i, j + 1] - Cn_old[i, j]))

        diff_Cn_down = 0 if j == 0 else ((Cn_old[i, j] - Cn_old[i, j - 1]))

        adv_right = (
            0
            if i == size_x - 1
            else (
                (Cn_old[i, j] * diff_Cb_right)
                if diff_Cb_right > 0
                else (Cn_old[i + 1, j] * diff_Cb_right)
            )
        )

        adv_left = (
            0
            if i == 0
            else (
                (Cn_old[i, j] * diff_Cb_left)
                if diff_Cb_left < 0
                else (Cn_old[i - 1, j] * diff_Cb_left)
            )
        )

        adv_up = (
            0
            if j == size_y - 1
            else (
                (Cn_old[i, j] * diff_Cb_up)
                if diff_Cb_up > 0
                else (Cn_old[i, j + 1] * diff_Cb_up)
            )
        )

        adv_down = (
            0
            if j == 0
            else (
                (Cn_old[i, j] * diff_Cb_down)
                if diff_Cb_down < 0
                else (Cn_old[i, j - 1] * diff_Cb_down)
            )
        )

        # Atualizando as concentrações de neutrófilos
        Cn_new[i][j] = (
            (k * Dn)
            / (h * h * phi)
            * (diff_Cn_right - diff_Cn_left + diff_Cn_up - diff_Cn_down)
            - (X_nb * k) / (h * h * phi) * (adv_right - adv_left + adv_up - adv_down)
            + (k / phi)
            * cu_fn(
                Cb_old, Cn_old, leu_source_points, i, j, y_n, Cn_max, lambd_bn, mi_n
            )
            + Cn_old[i, j]
        )

        # Armazenando os resultados para o passo de tempo atual
        Cb_final[time][i][j] = Cb_new[i][j]
        Cn_final[time][i][j] = Cn_new[i][j]

        grid.sync()
        