import numpy as np
import pickle as pk
import time
import matplotlib.pyplot as plt


class ProgBar:
    def __init__(self, n_elements, int_str):
        import sys

        self.n_elements = n_elements
        self.progress = 0

        print(int_str)

        # initiallizing progress bar

        info = "{:.2f}% - {:d} of {:d}".format(0, 0, n_elements)

        formated_bar = " " * int(50)

        sys.stdout.write("\r")

        sys.stdout.write("[%s] %s" % (formated_bar, info))

        sys.stdout.flush()

    def update(self, prog_info=None):
        import sys

        if prog_info == None:
            self.progress += 1

            percent = (self.progress) / self.n_elements * 100 / 2

            info = "{:.2f}% - {:d} of {:d}".format(
                percent * 2, self.progress, self.n_elements
            )

            formated_bar = "-" * int(percent) + " " * int(50 - percent)

            sys.stdout.write("\r")

            sys.stdout.write("[%s] %s" % (formated_bar, info))

            sys.stdout.flush()

        else:
            self.progress += 1

            percent = (self.progress) / self.n_elements * 100 / 2

            info = (
                "{:.2f}% - {:d} of {:d} ".format(
                    percent * 2, self.progress, self.n_elements
                )
                + prog_info
            )

            formated_bar = "-" * int(percent) + " " * int(50 - percent)

            sys.stdout.write("\r")

            sys.stdout.write("[%s] %s" % (formated_bar, info))

            sys.stdout.flush()


def preencher_matriz_uniforme(x_size, y_size):
    # Cria uma matriz de zeros com as dimensões fornecidas
    matriz = np.ones((x_size, y_size), dtype=int)

    return matriz


def preencher_matriz_radialmente(x_size, y_size):
    # Cria uma matriz de zeros com as dimensões fornecidas
    matriz = np.zeros((x_size, y_size), dtype=int)

    radius = 3
    cx, cy = (x_size // 2, y_size // 2)

    for i in range(x_size):
        for j in range(y_size):
            # Calculate distance from center to each point
            if (i - cx) ** 2 + (j - cy) ** 2 <= radius**2:
                matriz[i, j] = 1  # Set point inside the circle to 1

    return matriz


def preencher_matriz_randomicamente(x_size, y_size):

    # Cria uma matriz de zeros com as dimensões fornecidas
    matriz = np.zeros((x_size, y_size), dtype=int)

    # Calcula o número total de elementos a serem preenchidos com 1
    total_elementos = x_size * y_size  # <-- FIXED this. Was x_size * x_size

    elementos_para_preencher = int(0.08 * total_elementos)

    # Gera índices aleatórios únicos para preenchimento
    np.random.seed(42)
    indices = np.random.choice(total_elementos, elementos_para_preencher, replace=False)

    # Converte os índices lineares em índices matriciais
    for index in indices:
        i, j = divmod(index, y_size)
        matriz[i, j] = 1

    return matriz


def init_mesh(
    x_dom,
    y_dom,
    t_dom,
    h,
    k,
    center,
    radius,
    create_source=False,
    source_type="central",
    verbose=False,
):
    struct_name = (
        "h--"
        + str(h)
        + "__k--"
        + str(k)
        + "__x_dom_min--"
        + str(x_dom[0])
        + "__x_dom_max--"
        + str(x_dom[-1])
        + "__y_dom_min--"
        + "__t_dom_min--"
        + str(t_dom[0])
        + "__t_dom_max--"
        + str(t_dom[-1])
        + "__center--"
        + str(center)
        + "__radius--"
        + str(radius)
    )

    print("struct_name: ", struct_name)

    size_x = int(((x_dom[1] - x_dom[0]) / (h)))
    size_y = int(((y_dom[1] - y_dom[0]) / (h)))
    size_t = int(((t_dom[1] - t_dom[0]) / (k)) + 1)

    if create_source:
        if source_type == "central":
            leu_source_points = preencher_matriz_radialmente(size_x, size_y)
        elif source_type == "random":
            leu_source_points = preencher_matriz_randomicamente(size_x, size_y)
        elif source_type == "uniform":
            leu_source_points = preencher_matriz_uniforme(size_x, size_y)
        else:
            print("Not implemented type")
            return

        with open("source_points/lymph_vessels.pkl", "wb") as f:
            pk.dump(leu_source_points, f)

    else:
        with open("source_points/lymph_vessels.pkl", "rb") as f:
            leu_source_points = pk.load(f)

    print("Size x = {:d}, y = {:d} \n ".format(size_x, size_y))

    print(
        "Steps in time = {:d}\nSteps in space_x = {:d}\nSteps in space_y = {:d}\n".format(
            size_t,
            size_x,
            size_y,
        )
    )

    return (size_x, size_y, size_t, leu_source_points, struct_name)


def plot_results(size_t, size_x, t_dom, x_dom, Cb, Cn, leu_source_points):

    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32)
    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float32
    )

    # t_np, x_np, Cb, Cn, source_index already defined
    # source_index is assumed to be an array of x positions only (1D or Nx1)

    time_plot = np.linspace(0, size_t - 1, num=6, endpoint=True, dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    fig.suptitle(
        "$\\bf{Resposta\\ Imunológica}$ — Evolução em 1D", fontsize=18, weight="bold"
    )

    colors = plt.cm.viridis(np.linspace(0, 1, len(time_plot)))

    source_index = np.argwhere(leu_source_points[:, 0] == 1).ravel()

    # Plot Cb
    for i, time_inst in enumerate(time_plot):
        axes[0].plot(
            x_np,
            Cb[time_inst].squeeze(),
            label=f"t = {t_np[time_inst]:.2f}",
            color=colors[i],
            linewidth=2,
            alpha=0.85,
        )

    axes[0].scatter(
        x_np[source_index],  # assuming source_index is Nx2 still
        np.zeros(source_index.shape),  # put the markers at the top for visibility
        color="red",
        label="Fontes",
        s=40,
        marker="x",
    )

    axes[0].set_title("$C_p$ ao longo de x", fontsize=14)
    axes[0].set_ylabel("$C_p$", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot Cn
    for i, time_inst in enumerate(time_plot):
        axes[1].plot(
            x_np,
            Cn[time_inst].squeeze(),
            label=f"t = {t_np[time_inst]:.2f}",
            color=colors[i],
            linewidth=2,
            alpha=0.85,
        )

    axes[1].scatter(
        x_np[source_index],
        np.zeros((len(source_index))),
        color="red",
        label="Fontes",
        s=40,
        marker="x",
    )

    axes[1].set_title("$C_n$ ao longo de x", fontsize=14)
    axes[1].set_xlabel("x", fontsize=12)
    axes[1].set_ylabel("$C_n$", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
