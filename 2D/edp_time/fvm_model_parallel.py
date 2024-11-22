from numba import cuda


# Função que descreve a taxa de variação da concentração de bactérias (Cb)
@cuda.jit(device=True)
def cu_fb(Cb, Cn, i, j, cb, lambd_nb):
    # O crescimento de bactérias é reduzido pela presença de neutrófilos (Cn)
    # por um fator lambd_nb
    return (cb - lambd_nb * Cn[i, j]) * Cb[i, j]


# Função que descreve a taxa de variação da concentração de neutrófilos (Cn)
@cuda.jit(device=True)
def cu_fn(Cb, Cn, i, j, y_n, Cn_max, lambd_bn, mi_n):
    # Crescimento dos neutrófilos depende da presença de bactérias (Cb)
    # Também considera uma taxa de decaimento natural (mi_n) e a interação com
    # bactérias (lambd_bn)
    return (
        y_n * Cb[i, j] * (Cn_max - Cn[i, j])
        - lambd_bn * Cn[i, j] * Cb[i, j]
        - mi_n * Cn[i, j]
    )


# Função para aplicar condições iniciais à concentração de bactérias (Cb)
@cuda.jit(device=True)
def cu_apply_initial_conditions(Cb, tam_max, h):
    # Para uma região central do domínio, define uma concentração inicial de bactérias
    for i in range(tam_max):
        for j in range(tam_max):
            if (i * h >= 0.4) and (i * h) <= 0.6:
                if (j * h >= 0.4) and (j * h) <= 0.6:
                    Cb[i][j] = 0.2

    return Cb


# Função principal para resolver as equações diferenciais parciais usando diferenças finitas
@cuda.jit()
def cu_solve_pde(
    Cb_buf_0,
    Cn_buf_0,
    Cb_buf_1,
    Cn_buf_1,
    Cb_final,
    Cn_final,
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
):

    # Thread id in a 2D block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Block id in a 2D grid
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Block width, i.e. number of threads per block
    bwx = cuda.blockDim.x
    bwy = cuda.blockDim.y

    # Domain position
    i, j = cuda.grid(2)

    # print(
    #     "\nThread ids in a 2D block",
    #     tx,
    #     ty,
    #     "\nBlock ids in a 2D grid",
    #     bx,
    #     by,
    #     "\nBlock width, i.e. number of threads per block",
    #     bwx,
    #     bwy,
    #     "\nDomain position ",
    #     i,
    #     j,
    # )

    # Aplicando condições iniciais para a concentração de bactérias
    Cb_buf_0 = cu_apply_initial_conditions(Cb_buf_0, size_x, h)

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

        fx_Cb_right = 0 if (i == size_x - 1) else (Cb_old[i + 1, j] - Cb_old[i, j])

        fx_Cb_left = 0 if (i == 0) else (Cb_old[i, j] - Cb_old[i - 1, j])

        fx_Cb_up = 0 if (j == size_y - 1) else (Cb_old[i, j + 1] - Cb_old[i, j])

        fx_Cb_down = 0 if (j == 0) else (Cb_old[i, j] - Cb_old[i, j - 1])

        # Atualizando as concentrações de bactérias
        Cb_new[i][j] = (
            (k * Db)
            / (h * h * phi)
            * (fx_Cb_right * fx_Cb_right - fx_Cb_left + fx_Cb_up - fx_Cb_down)
            + (k / phi) * cu_fb(Cb_old, Cn_old, i, j, cb, lambd_nb)
            + Cb_old[i, j]
        )

        fx_Cn_right = 0 if i == size_x - 1 else ((Cn_old[i + 1, j] - Cn_old[i, j]))

        fx_Cn_left = 0 if i == 0 else ((Cn_old[i, j] - Cn_old[i - 1, j]))

        fx_Cn_up = 0 if j == size_y - 1 else ((Cn_old[i, j + 1] - Cn_old[i, j]))

        fx_Cn_down = 0 if j == 0 else ((Cn_old[i, j] - Cn_old[i, j - 1]))

        adv_right = (
            0
            if i == size_x - 1
            else (
                (Cn_old[i, j] * fx_Cb_right)
                if fx_Cb_right > 0
                else (Cn_old[i + 1, j] * fx_Cb_right)
            )
        )

        adv_left = (
            0
            if i == 0
            else (
                (Cn_old[i, j] * fx_Cb_left)
                if fx_Cb_left > 0
                else (Cn_old[i - 1, j] * fx_Cb_left)
            )
        )

        adv_up = (
            0
            if j == size_y - 1
            else (
                (Cn_old[i, j] * fx_Cb_up)
                if fx_Cb_up > 0
                else (Cn_old[i, j + 1] * fx_Cb_up)
            )
        )

        adv_down = (
            0
            if j == 0
            else (
                (Cn_old[i, j] * fx_Cb_down)
                if fx_Cb_down > 0
                else (Cn_old[i, j - 1] * fx_Cb_down)
            )
        )

        # Atualizando as concentrações de neutrófilos
        Cn_new[i][j] = (
            (k * Dn)
            / (h * h * phi)
            * (fx_Cn_right - fx_Cn_left + fx_Cn_up - fx_Cn_down)
            - (X_nb * k) / (h * h * phi) * (adv_right - adv_left + adv_up - adv_down)
            + (k / phi) * cu_fn(Cb_old, Cn_old, i, j, y_n, Cn_max, lambd_bn, mi_n)
            + Cn_old[i, j]
        )

        # Armazenando os resultados para o passo de tempo atual
        Cb_final[time][i][j] = Cb_new[i][j]
        Cn_final[time][i][j] = Cn_new[i][j]

        grid.sync()
