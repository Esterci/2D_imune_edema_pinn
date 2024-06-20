import numpy as np
import matplotlib.pyplot as plt

def fb(Cb, Cn, cb, lambd_nb):
    return (cb - lambd_nb * Cn) * Cb


def fn(Cb, Cn, Cn_max, y_n, lambd_bn, mi_n):
    return y_n * Cb * (Cn_max - Cn) - lambd_bn * Cn * Cb - mi_n * Cn


def fdm(
    k, phi, ksi, cb, Cn_max, lambd_nb, mi_n, lambd_bn, y_n, t_lower, t_upper, plot=False
):

    # Computing fdm model

    size_tt = int(((t_upper - t_lower) / (k))) + 1

    Cn_final = np.zeros((size_tt))
    Cb_final = np.zeros((size_tt))

    Cn_new = 0
    Cb_new = 0.2

    Cb_final[0] = Cb_new
    Cn_final[0] = Cn_new

    for time in range(1, size_tt):

        Cn_old = Cn_new
        Cb_old = Cb_new

        Cb_new = (k / phi) * (fb(Cb_old, Cn_old, cb, lambd_nb)) + Cb_old

        Cn_new = (k / phi) * (fn(Cb_old, Cn_old, Cn_max, y_n, lambd_bn, mi_n)) + Cn_old

        Cb_final[time] = Cb_new
        Cn_final[time] = Cn_new

    if plot:
        # Turn interactive plotting off
        plt.ioff()

        fig = plt.figure(figsize=[18, 9])

        fig.suptitle("Resposta imunológica a patógenos", fontsize=16)

        vmin = 0
        vmax = np.max([np.max(Cn_final), np.max(Cb_final)])

        # Plotango 3D
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(
            np.linspace(0, t_upper, num=len(Cb_final)),
            Cb_final,
            label="Con. de patógenos fdm",
        )
        ax.plot(
            np.linspace(0, t_upper, num=len(Cb_final)),
            Cn_final,
            label="Con. de leucócitos fdm",
        )
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Concentração")
        ax.set_ylim(vmin, vmax + 0.1)
        ax.legend()
        ax.grid()

        struct_name = (
            "k--"
            + str(k)
            + "__phi--"
            + str(phi)
            + "__ksi--"
            + str(ksi)
            + "__cb--"
            + str(cb)
            + "__Cn_max--"
            + str(Cn_max)
            + "__lambd_nb--"
            + str(lambd_nb)
            + "__mi_n--"
            + str(mi_n)
            + "__lambd_bn--"
            + str(lambd_bn)
            + "__y_n--"
            + str(y_n)
            + "__t_lower--"
            + str(t_lower)
            + "__t_upper--"
            + str(t_upper)
        )

        plt.savefig("edo_fdm_plot/" + struct_name + ".png")

        del fig

    return Cb_final, Cn_final


if __name__ == "__main__":
    k = 1.0
    t_lower = 0.0
    t_upper = 5.0
    phi = 0.2
    ksi = 0.0
    cb = 0.15
    C_nmax = 0.55
    mi_n = 0.2
    lambd_bn = 0.1
    y_n = 0.1

    Cp_old, Cl_old = fdm(
        k,
        phi,
        ksi,
        cb,
        C_nmax,
        1.8,
        mi_n,
        lambd_bn,
        y_n,
        t_lower,
        t_upper,
        plot="False",
    )
