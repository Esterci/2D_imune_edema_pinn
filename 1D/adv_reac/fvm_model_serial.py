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
def apply_initial_conditions(Cn, Cb, size_x, size_y):

    x = np.linspace(0, 1, num=size_x, endpoint=False)
    y = np.linspace(0, 1, num=size_y, endpoint=False)

    a = 0
    a2 = 1 - (1 / size_x)
    b = 4
    c = 2

    cb = np.exp(-(((x - a) * b) ** 2)) / c

    cn = np.exp(-(((x - a2) * b) ** 2)) / c

    Cn[:, 0] = cn
    Cb[:, 0] = cb

    return Cn, Cb


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

    Cn_new, Cb_new = apply_initial_conditions(Cn_new, Cb_new, size_x, size_y)

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

                vx_right = X_nb * diff_Cb_right / h
                vx_left = X_nb * diff_Cb_left / h
                vy_up = X_nb * diff_Cb_up / h
                vy_down = X_nb * diff_Cb_down / h

                max_vx = max(abs(vx_left), abs(vx_right))
                max_vy = max(abs(vy_down), abs(vy_up))
                max_v = max(max_vx, max_vy)

                if max_v > global_max_v:
                    global_max_v = max_v

                # Atualizando as concentrações de bactérias
                #                Cb_new[i][j] = (
                #                    (k * Db)
                #                    / (h * h * phi)
                #                    * (diff_Cb_right - diff_Cb_left + diff_Cb_up - diff_Cb_down)
                #                    + (k / phi) * fb(Cb_old, Cn_old, i, j, cb, lambd_nb)
                #                    + Cb_old[i, j]
                #                )

                Cb_new[i][j] = Cb_old[i, j]

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
                        vx_right * (Cn_old[i, j] if vx_right > 0 else Cn_old[i + 1, j])
                    )
                )

                adv_left = (
                    0
                    if i == 0
                    else (vx_left * (Cn_old[i - 1, j] if vx_left > 0 else Cn_old[i, j]))
                )

                adv_up = (
                    0
                    if j == size_y - 1
                    else (vy_up * (Cn_old[i, j] if vy_up > 0 else Cn_old[i, j + 1]))
                )

                adv_down = (
                    0
                    if j == 0
                    else (vy_down * (Cn_old[i, j - 1] if vy_down > 0 else Cn_old[i, j]))
                )

                # Atualizando as concentrações de neutrófilos
                Cn_new[i][j] = Cn_new[i][j] = Cn_old[i, j] - (k / (h * phi)) * (
                    adv_right - adv_left + adv_up - adv_down
                )

                # Armazenando os resultados para o passo de tempo atual
                Cb_final[time][i][j] = Cb_new[i][j]
                Cn_final[time][i][j] = Cn_new[i][j]

        # Calcula critério de CFL

        cfl_adv = global_max_v * k / h

        cfl_dif = np.max((2 * Db * k / (h * h), 2 * Dn * k / (h * h)))

        if cfl_adv + cfl_dif > 1:
            print(
                "ERROR - CFL criterium not matched on iteration {}: {}".format(
                    time, cfl_adv + cfl_dif
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
