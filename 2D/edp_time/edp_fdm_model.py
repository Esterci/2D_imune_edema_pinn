import numpy as np
import sys
import argparse
import pickle as pk
import time


def fb(Cb, Cn, cb, lambd_nb):
    return (cb - lambd_nb * Cn) * Cb


def fn(Cb, Cn, Cn_max, y_n, lambd_bn, mi_n):
    return y_n * Cb * (Cn_max - Cn) - lambd_bn * Cn * Cb - mi_n * Cn


def phix(Cbipj, Cbimj, Cbij, i, j, tam_max):
    ax = (Cbipj - Cbimj) / (2 * h)

    if (i == 0 or i == tam_max - 1) or (j == 0 or j == tam_max - 1) or (ax == 0):
        return 0

    elif ax > 0:
        return (ax * k / h) * (Cbij - Cbimj)

    elif ax < 0:
        return (ax * k / h) * (Cbipj - Cbij)


def phiy(Cbijp, Cbijm, Cbij, i, j, tam_max):
    ax = (Cbijp - Cbijm) / (2 * h)

    if (i == 0 or i == tam_max - 1) or (j == 0 or j == tam_max - 1) or (ax == 0):
        return 0

    elif ax > 0:
        return (ax * k / h) * (Cbij - Cbijm)

    elif ax < 0:
        return (ax * k / h) * (Cbijp - Cbij)


def apply_initial_conditions(Cb, tam_max):
    for i in range(tam_max):
        for j in range(tam_max):
            if (i * h >= 0.4) and (i * h) <= 0.6:
                if (j * h >= 0.4) and (j * h) <= 0.6:
                    Cb[i][j] = 0.2

    return Cb


def solve_pde(
    h,
    k,
    Db,
    Dn,
    phi,
    ksi,
    cb,
    lambd_nb,
    mi_n,
    lambd_bn,
    y_n,
    Cn_max,
    size_x,
    size_y,
    size_tt,
):

    Cn_new = np.zeros((size_x, size_y))
    Cb_new = np.zeros((size_x, size_y))

    Cn_final = np.zeros((size_tt, size_x, size_y))
    Cb_final = np.zeros((size_tt, size_x, size_y))

    Cb_new = apply_initial_conditions(Cb_new, size_x)

    Cb_final[0] = Cb_new
    Cn_final[0] = Cn_new

    u_a = u_b = u_c = u_d = 0

    r_n = (Dn * k) / (phi * (h * h))
    r_b = (Db * k) / (phi * (h * h))

    for time in range(size_tt):
        Cn_old = Cn_new.copy()
        Cb_old = Cb_new.copy()
        for i in range(size_x):
            for j in range(size_y):
                # Tratando Cb
                if j == size_y - 1:
                    Cb_uijp = 2 * h * u_c + Cb_old[i][size_y - 1]

                else:
                    Cb_uijp = Cb_old[i][j + 1]
                if j == 0:
                    Cb_uijm = 2 * h * u_a + Cb_old[i][1]

                else:
                    Cb_uijm = Cb_old[i][j - 1]

                if i == size_x - 1:
                    Cb_uipj = 2 * h * u_d + Cb_old[size_x - 1][j]
                else:
                    Cb_uipj = Cb_old[i + 1][j]
                if i == 0:
                    Cb_uimj = 2 * h * u_b * Cb_old[1][j]
                else:
                    Cb_uimj = Cb_old[i - 1][j]

                # Tratando Cn
                if j == size_y - 1:
                    Cn_uijp = 2 * h * u_c + Cn_old[i][size_y - 1]
                else:
                    Cn_uijp = Cn_old[i][j + 1]
                #
                if j == 0:
                    Cn_uijm = 2 * h * u_a + Cn_old[i][1]
                else:
                    Cn_uijm = Cn_old[i][j - 1]
                #
                if i == size_x - 1:
                    Cn_uipj = 2 * h * u_d + Cn_old[size_x - 1][j]
                else:
                    Cn_uipj = Cn_old[i + 1][j]
                #
                if i == 0:
                    Cn_uimj = 2 * h * u_b * Cn_old[1][j]
                else:
                    Cn_uimj = Cn_old[i - 1][j]

                Cb_new[i][j] = (
                    r_b * (Cb_uimj + Cb_uipj - 4 * Cb_old[i][j] + Cb_uijp + Cb_uijm)
                    + (k / phi) * (fb(Cb_old[i][j], Cn_old[i][j], cb, lambd_nb))
                    + Cb_old[i][j]
                )

                Cn_new[i][j] = (
                    r_n * (Cn_uimj + Cn_uipj - 4 * Cn_old[i][j] + Cn_uijp + Cn_uijm)
                    + (k / phi)
                    * (fn(Cb_old[i][j], Cn_old[i][j], Cn_max, y_n, lambd_bn, mi_n))
                    + Cn_old[i][j]
                    + ksi
                    * (
                        -phix(Cb_uipj, Cb_uimj, Cb_old[i][j], i, j, size_x)
                        - phiy(Cb_uijp, Cb_uijm, Cb_old[i][j], i, j, size_y)
                    )
                )

                Cb_final[time][i][j] = Cb_new[i][j]
                Cn_final[time][i][j] = Cn_new[i][j]

    return Cb_final, Cn_final


if __name__ == "__main__":
    # Parsing model parameters

    parser = argparse.ArgumentParser(description="", add_help=False)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--n_iterations",
        type=int,
        action="store",
        dest="iterations",
        required=True,
        default=None,
        help="",
    )

    args = parser.parse_args()

    args_dict = vars(args)

    h = 0.1
    k = 0.1
    Db = 0.0001
    Dn = 0.0001
    phi = 0.2
    ksi = 0.0
    cb = 0.15
    lambd_nb = 1.8
    mi_n = 0.2
    lambd_bn = 0.1
    y_n = 0.1
    Cn_max = 0.5
    X_nb = 1e-4
    x_dom = (0, 1)
    y_dom = (0, 1)
    t_dom = (0, 10)

    size_x = int(((x_dom[1] - x_dom[0]) / (h))) + 1
    size_y = int(((y_dom[1] - y_dom[0]) / (h))) + 1
    size_tt = int(((t_dom[1] - t_dom[0]) / (k))) + 1

    print("Size x = {:d}, y = {:d} \n ".format(size_x, size_y))

    print(
        "Steps in time = {:d}\nSteps in space_x = {:d}\nSteps in space_y = {:d}\n".format(
            size_tt,
            size_x,
            size_y,
        )
    )

    CFL = (Db * k) / ((2 * (h * h)))

    print("CFL: ", CFL, "\n")

    if CFL >= 1:
        print("Criterio CFL não satisfeito\n")

        sys.exit(400)

    fdm_time = []

    for i in range(args_dict["iterations"]):

        start = time.time()

        Cb, Cn = solve_pde(
            h,
            k,
            Db,
            Dn,
            phi,
            ksi,
            cb,
            lambd_nb,
            mi_n,
            lambd_bn,
            y_n,
            Cn_max,
            size_x,
            size_y,
            size_tt,
        )

        end = time.time()

        run_time = end - start

        fdm_time.append(run_time)

    struct_name = (
        "h--"
        + str(h)
        + "__k--"
        + str(k)
        + "__Db--"
        + str(Db)
        + "__Dn--"
        + str(Dn)
        + "__phi--"
        + str(phi)
        + "__ksi--"
        + str(ksi)
        + "__cb--"
        + str(cb)
        + "__lambd_nb--"
        + str(lambd_nb)
        + "__mi_n--"
        + str(mi_n)
        + "__lambd_bn--"
        + str(lambd_bn)
        + "__y_n--"
        + str(y_n)
        + "__Cn_max--"
        + str(Cn_max)
        + "__X_nb--"
        + str(X_nb)
        + "__x_dom_min--"
        + str(x_dom[0])
        + "__x_dom_max--"
        + str(x_dom[-1])
        + "__y_dom_min--"
        + str(y_dom[0])
        + "__y_dom_max--"
        + str(y_dom[-1])
        + "__t_dom_min--"
        + str(t_dom[0])
        + "__t_dom_max--"
        + str(t_dom[-1])
    )

    print("struct_name: ", struct_name)

    with open("fdm_sim/Cp__" + struct_name + ".pkl", "wb") as f:
        pk.dump(Cb, f)

    with open("fdm_sim/Cl__" + struct_name + ".pkl", "wb") as f:
        pk.dump(Cn, f)

    with open("fdm_sim/time__" + struct_name + ".pkl", "wb") as f:
        pk.dump(fdm_time, f)
