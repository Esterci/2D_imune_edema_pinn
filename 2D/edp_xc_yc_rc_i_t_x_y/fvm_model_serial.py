import numpy as np


# Função que descreve a taxa de variação da concentração de bactérias (Cb)
def fb(Cb, Cn, i, j, cb, lambd_nb):
    # O crescimento de bactérias é reduzido pela presença de neutrófilos (Cn)
    # por um fator lambd_nb
    return (cb - lambd_nb * Cn[i, j]) * Cb[i, j]


# Função que descreve a taxa de variação da concentração de neutrófilos (Cn)
def fn(Cb, Cn, source_points, i, j, y_n, Cn_max, lambd_bn, mi_n):
    # Crescimento dos neutrófilos depende da presença de bactérias (Cb)
    # Também considera uma taxa de decaimento natural (mi_n) e a interação com
    # bactérias (lambd_bn)
    return (
        y_n * Cb[i, j] * (Cn_max - Cn[i, j]) * source_points[i, j]
        - lambd_bn * Cn[i, j] * Cb[i, j]
        - mi_n * Cn[i, j]
    )


# Função para aplicar condições iniciais à concentração de bactérias (Cb)
def apply_initial_conditions(ini_cond, Cb, cx, cy, radius, tam_max):
    for i in range(tam_max):
        for j in range(tam_max):
            # Calculate distance from center to each point
            if (i - cx) ** 2 + (j - cy) ** 2 <= radius**2:
                Cb[i][j] = ini_cond  # Set point inside the circle to 1

    return Cb


# Função principal para resolver as equações diferenciais parciais usando diferenças finitas
def solve_pde(
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
    verbose=False,
):

    # Inicializando as matrizes para concentrações de neutrófilos (Cn) e bactérias (Cb)
    Cn_new = np.zeros((size_x, size_y))
    Cb_new = np.zeros((size_x, size_y))

    # Matrizes para armazenar as concentrações em cada passo de tempo
    Cn_final = np.zeros((size_t, size_x, size_y))
    Cb_final = np.zeros((size_t, size_x, size_y))

    # Aplicando condições iniciais para a concentração de bactérias
    cx_real, cy_real = center
    cx_disc = cx_real / h
    cy_disc = cy_real / h
    radius_disc = radius / h

    Cb_new = apply_initial_conditions(
        initial_cond, Cb_new, cx_disc, cy_disc, radius_disc, size_x
    )

    # Armazenando as condições iniciais
    Cb_final[0] = Cb_new
    Cn_final[0] = Cn_new

    # Loop sobre o tempo
    for time in range(1, size_t):
        # Atualizando as concentrações anteriores (passo temporal anterior)
        Cn_old = Cn_new.copy()
        Cb_old = Cb_new.copy()

        global_max_v = 0

        # Loop sobre o espaço (malha espacial)
        for i in range(size_x):

            for j in range(size_y):

                diff_Cb_right = (
                    0 if (i == size_x - 1) else (Cb_old[i + 1, j] - Cb_old[i, j])
                )

                diff_Cb_left = 0 if (i == 0) else (Cb_old[i, j] - Cb_old[i - 1, j])

                diff_Cb_up = (
                    0 if (j == size_y - 1) else (Cb_old[i, j + 1] - Cb_old[i, j])
                )

                diff_Cb_down = 0 if (j == 0) else (Cb_old[i, j] - Cb_old[i, j - 1])

                max_vx = np.max(
                    (
                        abs(diff_Cb_left * X_nb / h),
                        abs(diff_Cb_right * X_nb / h),
                    )
                )

                max_vy = np.max(
                    (
                        abs(diff_Cb_down * X_nb / h),
                        abs(diff_Cb_up * X_nb / h),
                    )
                )

                max_v = max(max_vx, max_vy)

                if max_v > global_max_v:
                    global_max_v = max_v

                # Atualizando as concentrações de bactérias
                Cb_new[i][j] = (
                    (k * Db)
                    / (h * h * phi)
                    * (diff_Cb_right - diff_Cb_left + diff_Cb_up - diff_Cb_down)
                    + (k / phi) * fb(Cb_old, Cn_old, i, j, cb, lambd_nb)
                    + Cb_old[i, j]
                )

                diff_Cn_right = (
                    0 if i == size_x - 1 else ((Cn_old[i + 1, j] - Cn_old[i, j]))
                )

                diff_Cn_left = 0 if i == 0 else ((Cn_old[i, j] - Cn_old[i - 1, j]))

                diff_Cn_up = (
                    0 if j == size_y - 1 else ((Cn_old[i, j + 1] - Cn_old[i, j]))
                )

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
                    - (X_nb * k)
                    / (h * h * phi)
                    * (adv_right - adv_left + adv_up - adv_down)
                    + (k / phi)
                    * fn(
                        Cb_old,
                        Cn_old,
                        leu_source_points,
                        i,
                        j,
                        y_n,
                        Cn_max,
                        lambd_bn,
                        mi_n,
                    )
                    + Cn_old[i, j]
                )

                # Armazenando os resultados para o passo de tempo atual
                Cb_final[time][i][j] = Cb_new[i][j]
                Cn_final[time][i][j] = Cn_new[i][j]

        # Calcula critério de CFL

        cfl_adv = global_max_v * k / h

        cfl_dif = np.max((4 * Db * k / (h * h), 4 * Dn * k / (h * h)))

        if cfl_adv > 1:
            print(
                "ERROR - CFL criterium not matched on iteration {} for advection: {}".format(
                    time, cfl_adv
                )
            )
            break

        elif cfl_dif > 1:
            print(
                "ERROR - CFL criterium not matched on iteration {} for difusion: {}".format(
                    time, cfl_dif
                )
            )
            break

        else:
            if (time % (size_t // 10) == 0 or time == 0) and verbose:
                print(
                    "CFL criterium on iteration {}: {}".format(
                        time, max(cfl_adv, cfl_dif)
                    )
                )

    # Retornando as matrizes finais de concentração de bactérias e neutrófilos ao
    # longo do tempo
    return Cb_final, Cn_final
