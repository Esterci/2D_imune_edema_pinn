import numpy as np
import pickle as pk
import time
import matplotlib.pyplot as plt


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


def preencher_matriz_randomicamente(x_size, y_size, x_dom, percent, source_behavior):

    # Cria uma matriz de zeros com as dimensões fornecidas
    matriz = np.zeros((x_size, y_size), dtype=float)

    coordinates = np.zeros((x_size, y_size), dtype=float)

    # Calcula o número total de elementos a serem preenchidos com 1
    total_elementos = x_size * y_size  # <-- FIXED this. Was x_size * x_size

    elementos_para_preencher = int(percent * total_elementos)

    # Gera índices aleatórios únicos para preenchimento

    indices = np.random.choice(total_elementos, elementos_para_preencher, replace=False)

    if source_behavior == "boolean":

        # Converte os índices lineares em índices matriciais
        for index in indices:
            i, j = divmod(index, y_size)
            matriz[i, j] = 1

        return matriz

    elif source_behavior == "gaussian":

        x_np = np.linspace(
            x_dom[0], x_dom[-1], num=x_size, endpoint=False, dtype=np.float32
        )

        # Converte os índices lineares em índices matriciais
        for index in indices:
            i, j = divmod(index, y_size)
            x_center = x_np[i]

            gaussian = np.exp(-(((x_np - x_center) * 60) ** 2)) / 2

            coordinates[i, j] = 1

            matriz[:, j] += gaussian

        return (matriz, coordinates)

    else:
        print("Source type not implemented")

        return


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
    source_behavior="boolean",
    percent=0.2,
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
            leu_source_points = preencher_matriz_randomicamente(
                size_x, size_y, x_dom, percent, source_behavior
            )
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


import matplotlib.animation as animation


def plot_comparison(
    size_t, size_x, t_dom, x_dom, Cb, Cn, Cb_pinn, Cn_pinn, leu_source_points
):

    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32)
    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float32
    )

    # t_np, x_np, Cb, Cn, source_index already defined
    # source_index is assumed to be an array of x positions only (1D or Nx1)

    time_plot = np.linspace(0, size_t - 1, num=6, endpoint=True, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    fig.suptitle(
        "$\\bf{Resposta\\ Imunológica}$ — Evolução em 1D", fontsize=18, weight="bold"
    )

    colors = plt.cm.viridis(np.linspace(0, 1, len(time_plot)))

    source_index = np.argwhere(leu_source_points[:, 0] == 1).ravel()

    # Plot Cb
    for i, time_inst in enumerate(time_plot):
        axes[0, 0].plot(
            x_np,
            Cb[time_inst].squeeze(),
            label=f"t = {t_np[time_inst]:.2f}",
            color=colors[i],
            linewidth=2,
            alpha=0.85,
        )

    axes[0, 0].scatter(
        x_np[source_index],  # assuming source_index is Nx2 still
        np.zeros(source_index.shape),  # put the markers at the top for visibility
        color="red",
        label="Fontes",
        s=40,
        marker="x",
    )

    axes[0, 0].set_title("$C_p$ ao longo de x, FVM", fontsize=14)
    axes[0, 0].set_ylabel("$C_p$", fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Plot Cn
    for i, time_inst in enumerate(time_plot):
        axes[0, 1].plot(
            x_np,
            Cn[time_inst].squeeze(),
            label=f"t = {t_np[time_inst]:.2f}",
            color=colors[i],
            linewidth=2,
            alpha=0.85,
        )

    axes[0, 1].scatter(
        x_np[source_index],
        np.zeros((len(source_index))),
        color="red",
        label="Fontes",
        s=40,
        marker="x",
    )

    axes[0, 1].set_title("$C_n$ ao longo de x, FVM", fontsize=14)
    axes[0, 1].set_xlabel("x", fontsize=12)
    axes[0, 1].set_ylabel("$C_n$", fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Plot Cb
    for i, time_inst in enumerate(time_plot):
        axes[1, 0].plot(
            x_np,
            Cb_pinn[time_inst].squeeze(),
            label=f"t = {t_np[time_inst]:.2f}",
            color=colors[i],
            linewidth=2,
            alpha=0.85,
        )

    axes[1, 0].scatter(
        x_np[source_index],  # assuming source_index is Nx2 still
        np.zeros(source_index.shape),  # put the markers at the top for visibility
        color="red",
        label="Fontes",
        s=40,
        marker="x",
    )

    axes[1, 0].set_title("$C_p$ ao longo de x, FVM", fontsize=14)
    axes[1, 0].set_ylabel("$C_p$", fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Plot Cn
    for i, time_inst in enumerate(time_plot):
        axes[1, 1].plot(
            x_np,
            Cn_pinn[time_inst].squeeze(),
            label=f"t = {t_np[time_inst]:.2f}",
            color=colors[i],
            linewidth=2,
            alpha=0.85,
        )

    axes[1, 1].scatter(
        x_np[source_index],
        np.zeros((len(source_index))),
        color="red",
        label="Fontes",
        s=40,
        marker="x",
    )

    axes[1, 1].set_title("$C_n$ ao longo de x, FVM", fontsize=14)
    axes[1, 1].set_xlabel("x", fontsize=12)
    axes[1, 1].set_ylabel("$C_n$", fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def animate_1D_comparison(
    size_t,
    size_x,
    t_dom,
    x_dom,
    Cb,
    Cn,
    Cb_pinn,
    Cn_pinn,
    leu_source_points,
    delta_t,
    frame_time,
    name="evolucao_1D",
    show=False,
):
    """
    Gera uma animação que mostra a evolução de Cp e Cn em 1D ao longo do tempo.
    Parametros:
        size_t, size_x: nº de pontos no tempo e espaço
        t_dom, x_dom: domínios (lista ou tupla [min, max] para tempo e espaço)
        Cb, Cn: arrays de shape (size_t, size_x) com valores de Cp e Cn
        leu_source_points: array booleano (ou 0/1) indicando onde há fontes (mesmo shape de x ou Nx1).
    """

    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32)
    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float32
    )

    # Converte leu_source_points para índices (caso seja um array Nx1 de 0/1)
    # Se já estiver pronto, pode ajustar conforme sua lógica
    source_index = np.argwhere(leu_source_points[:, 0] == 1).ravel()

    # Cria figura e eixos
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "$\\bf{Resposta\\ Imunológica}$ — Evolução em 1D", fontsize=18, weight="bold"
    )

    # Primeiro subplot: Cp
    ax_cp = axes[0, 0]
    (line_cp,) = ax_cp.plot([], [], "b-", lw=2, alpha=0.85, label="Cp")
    ax_cp.set_ylabel("$C_p$")
    ax_cp.set_title("$C_p$ ao longo de x, FVM", fontsize=14)
    ax_cp.grid(True, linestyle="--", alpha=0.5)
    ax_cp.set_xlim(0, 1)
    ax_cp.set_ylim(0, np.max(Cb) * 1.1)
    ax_cp.legend()

    # Segundo subplot: Cn
    ax_cn = axes[0, 1]
    (line_cn,) = ax_cn.plot([], [], "g-", lw=2, alpha=0.85, label="Cn")
    sc_cn = ax_cn.scatter(
        x_np[source_index],
        np.zeros_like(source_index),
        color="red",
        label="Fontes",
        s=40,
        marker="x",
    )
    ax_cn.set_xlabel("x", fontsize=12)
    ax_cn.set_ylabel("$C_n$")
    ax_cn.set_title("$C_n$ ao longo de x, FVM", fontsize=14)
    ax_cn.grid(True, linestyle="--", alpha=0.5)
    ax_cn.set_xlim(0, 1)
    ax_cn.set_ylim(0, np.max(Cn) * 1.1)
    ax_cn.legend()

    # Primeiro subplot: Cp_pinn
    ax_cp_pinn = axes[1, 0]
    (line_cp_pinn,) = ax_cp_pinn.plot([], [], "b-", lw=2, alpha=0.85, label="Cp_pinn")
    ax_cp_pinn.set_ylabel("$C_p$")
    ax_cp_pinn.set_title("$C_p$ ao longo de x, PINN", fontsize=14)
    ax_cp_pinn.grid(True, linestyle="--", alpha=0.5)
    ax_cp_pinn.set_xlim(0, 1)
    ax_cp_pinn.set_ylim(0, np.max(Cb) * 1.1)
    ax_cp_pinn.legend()

    # Segundo subplot: Cn_pinn
    ax_cn_pinn = axes[1, 1]
    (line_cn_pinn,) = ax_cn_pinn.plot([], [], "g-", lw=2, alpha=0.85, label="Cn_pinn")
    sc_cn_pinn = ax_cn_pinn.scatter(
        x_np[source_index],
        np.zeros_like(source_index),
        color="red",
        label="Fontes",
        s=40,
        marker="x",
    )
    ax_cn_pinn.set_xlabel("x", fontsize=12)
    ax_cn_pinn.set_ylabel("$C_n$")
    ax_cn_pinn.set_title("$C_n$ ao longo de x, PINN", fontsize=14)
    ax_cn_pinn.grid(True, linestyle="--", alpha=0.5)
    ax_cn_pinn.set_xlim(0, 1)
    ax_cn_pinn.set_ylim(0, np.max(Cn) * 1.1)
    ax_cn_pinn.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Função para inicializar a animação
    def init():
        line_cp.set_data([], [])
        line_cn.set_data([], [])
        line_cp_pinn.set_data([], [])
        line_cn_pinn.set_data([], [])
        return line_cp, line_cn, line_cp_pinn, line_cn_pinn, sc_cn, sc_cn_pinn

    # Função de atualização a cada frame
    def update(frame):
        # frame varia de 0 até size_t-1
        cp_vals = Cb[frame].squeeze()  # shape (size_x,)
        cn_vals = Cn[frame].squeeze()
        cp_vals_pinn = Cb_pinn[frame].squeeze()  # shape (size_x,)
        cn_vals_pinn = Cn_pinn[frame].squeeze()

        # Atualiza as linhas
        line_cp.set_data(x_np, cp_vals)
        line_cn.set_data(x_np, cn_vals)
        line_cp_pinn.set_data(x_np, cp_vals_pinn)
        line_cn_pinn.set_data(x_np, cn_vals_pinn)

        # Atualiza posições dos scatters se necessário (aqui, continua em 0)
        sc_cn.set_offsets(
            np.column_stack((x_np[source_index], np.zeros_like(source_index)))
        )

        sc_cn_pinn.set_offsets(
            np.column_stack((x_np[source_index], np.zeros_like(source_index)))
        )

        # Ajusta título (opcional) para mostrar tempo
        ax_cp.set_title(f"$C_p$ ao longo de x, t = {t_np[frame]:.2f}")
        ax_cn.set_title(f"$C_n$ ao longo de x, t = {t_np[frame]:.2f}")

        return line_cp, line_cn, line_cp_pinn, line_cn_pinn, sc_cn

    frames_indices = range(0, size_t, delta_t)

    # Cria animação
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames_indices,  # total de frames = nº de instantes de tempo
        init_func=init,
        blit=True,  # se quiser performance, use True, mas requer cuidado
        interval=frame_time,  # interval em ms entre frames
    )

    # Salvar como vídeo MP4 (necessita FFmpeg instalado):

    if show:
        plt.show()

    ani.save("fvm_animations/" + name + ".mp4", fps=5)
