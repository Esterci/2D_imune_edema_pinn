import numpy as np
import pickle as pk
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_results(size_t, size_x, t_dom, x_dom, Cb, Cn):

    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32)
    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float32
    )

    time_plot = np.linspace(0, size_t - 1, num=6, endpoint=True, dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    fig.suptitle(
        "$\\bf{Resposta\\ Imunológica}$ — Evolução em 1D", fontsize=18, weight="bold"
    )

    colors = plt.cm.viridis(np.linspace(0, 1, len(time_plot)))

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

    axes[1].set_title("$C_n$ ao longo de x", fontsize=14)
    axes[1].set_xlabel("x", fontsize=12)
    axes[1].set_ylabel("$C_n$", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()
    plt.savefig("fvm.png")


def plot_comparison(size_t, size_x, t_dom, x_dom, Cb, Cn, Cb_pinn, Cn_pinn):

    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32)
    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float32
    )

    time_plot = np.linspace(0, size_t - 1, num=6, endpoint=True, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    fig.suptitle(
        "$\\bf{Resposta\\ Imunológica}$ — Evolução em 1D", fontsize=18, weight="bold"
    )

    colors = plt.cm.viridis(np.linspace(0, 1, len(time_plot)))

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

    axes[1, 0].set_title("$C_p$ ao longo de x, PINN", fontsize=14)
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

    axes[1, 1].set_title("$C_n$ ao longo de x, PINN", fontsize=14)
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
    """

    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32)
    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float32
    )

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


def plot_comparison_pinn(
    size_t,
    size_x,
    t_dom,
    x_dom,
    Cb,
    Cn,
    Cb_pinn,
    Cn_pinn,
    Cb_nn,
    Cn_nn,
):
    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32)
    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float32
    )
    time_plot = np.linspace(0, size_t - 1, num=6, endpoint=True, dtype=int)

    fig, axes = plt.subplots(3, 2, figsize=(6, 3), sharex=True)
    # fig.suptitle(
    #     "$\\bf{Resposta\\ Imunológica}$ — Evolução em 1D", fontsize=18, weight="bold"
    # )

    colors = plt.cm.viridis(np.linspace(0, 1, len(time_plot)))

    titles = [
        [
            "$C_p$ MVF",
            "$C_l$ MVF",
        ],
        [
            "$C_p$ PINN",
            "$C_l$ PINN",
        ],
        [
            "$C_p$ RN",
            "$C_l$ RN",
        ],
    ]

    for row, (Cb_data, Cn_data) in enumerate(
        [(Cb, Cn), (Cb_pinn, Cn_pinn), (Cb_nn, Cn_nn)]
    ):
        for col, (data, ylabel) in enumerate(
            zip([Cb_data, Cn_data], ["$C_p$", "$C_n$"])
        ):
            ax = axes[row, col]
            for i, time_inst in enumerate(time_plot):
                ax.plot(
                    x_np,
                    data[time_inst].squeeze(),
                    label=f"t = {t_np[time_inst]:.2f}",
                    color=colors[i],
                    linewidth=2,
                    alpha=0.85,
                )

            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(titles[row][col], fontsize=14)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_comparison_contour(size_t, size_x, t_dom, x_dom, Cb, Cn, Cb_comp, Cn_comp):
    # Domínios
    t = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32)
    x = np.linspace(x_dom[0], x_dom[-1], num=size_x, endpoint=True, dtype=np.float32)
    X, T = np.meshgrid(x, t)

    # Lista de dados para os 6 plots
    data_list = [
        Cb,
        Cb_comp,
        np.abs(Cb - Cb_comp),
        Cn,
        Cn_comp,
        np.abs(Cn - Cn_comp),
    ]

    # Plotando cada gráfico individualmente
    for i, data in enumerate(data_list):
        fig, ax = plt.subplots(figsize=(6, 4))
        Z = data.reshape(size_t, size_x)
        vmin = np.min(Z)
        vmax = np.max(Z)

        contour = ax.contourf(X, T, Z, cmap="jet", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin, vmax, num=5))

        ax.set_xlabel("x")
        ax.set_ylabel("t")
        # Título removido
        ax.grid(False)
        plt.tight_layout()
        plt.show()
