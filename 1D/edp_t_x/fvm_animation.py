import matplotlib.pyplot as plt
import json
from glob import glob
from pinn import *
import pickle as pk
import matplotlib.animation as animation

import matplotlib.animation as animation


def animate_1D_evolution(
    size_t,
    size_x,
    t_dom,
    x_dom,
    Cb,
    Cn,
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
    x_np = np.linspace(x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float32)

    # Converte leu_source_points para índices (caso seja um array Nx1 de 0/1)
    # Se já estiver pronto, pode ajustar conforme sua lógica
    source_index = np.argwhere(leu_source_points[:, 0] == 1).ravel()

    # Cria figura e eixos
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "$\\bf{Resposta\\ Imunológica}$ — Evolução em 1D", fontsize=18, weight="bold"
    )

    # Primeiro subplot: Cp
    ax_cp = axes[0]
    (line_cp,) = ax_cp.plot([], [], "b-", lw=2, alpha=0.85, label="Cp")
    ax_cp.set_ylabel("$C_p$")
    ax_cp.set_title("$C_p$ ao longo de x", fontsize=14)
    ax_cp.grid(True, linestyle="--", alpha=0.5)
    ax_cp.set_xlim(0, 1)
    ax_cp.set_ylim(0, np.max(Cb) * 1.1)
    ax_cp.legend()

    # Segundo subplot: Cn
    ax_cn = axes[1]
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
    ax_cn.set_title("$C_n$ ao longo de x", fontsize=14)
    ax_cn.grid(True, linestyle="--", alpha=0.5)
    ax_cn.set_xlim(0, 1)
    ax_cn.set_ylim(0, np.max(Cn) * 1.1)
    ax_cn.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Função para inicializar a animação
    def init():
        line_cp.set_data([], [])
        line_cn.set_data([], [])
        return line_cp, line_cn, sc_cn

    # Função de atualização a cada frame
    def update(frame):
        # frame varia de 0 até size_t-1
        cp_vals = Cb[frame].squeeze()  # shape (size_x,)
        cn_vals = Cn[frame].squeeze()

        # Atualiza as linhas
        line_cp.set_data(x_np, cp_vals)
        line_cn.set_data(x_np, cn_vals)
        # Atualiza posições dos scatters se necessário (aqui, continua em 0)
        sc_cn.set_offsets(
            np.column_stack((x_np[source_index], np.zeros_like(source_index)))
        )

        # Ajusta título (opcional) para mostrar tempo
        ax_cp.set_title(f"$C_p$ ao longo de x, t = {t_np[frame]:.2f}")
        ax_cn.set_title(f"$C_n$ ao longo de x, t = {t_np[frame]:.2f}")

        return line_cp, line_cn, sc_cn

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


# Exemplo de uso (assumindo que Cb.shape = (size_t, size_x) etc.):
# animate_1D_evolution(size_t, size_x, t_dom, x_dom, Cb, Cn, leu_source_points)

if __name__ == "__main__":

    # Opening JSON file
    with open("control_dicts/constant_properties.json", "r") as openfile:
        # Reading from json file
        constant_properties = json.load(openfile)

    Db = constant_properties["Db"]
    Dn = constant_properties["Dn"]
    phi = constant_properties["phi"]
    cb = constant_properties["cb"]
    lambd_nb = constant_properties["lambd_nb"]
    mi_n = constant_properties["mi_n"]
    lambd_bn = constant_properties["lambd_bn"]
    y_n = constant_properties["y_n"]
    Cn_max = constant_properties["Cn_max"]
    X_nb = constant_properties["X_nb"]
    central_ini_cond = constant_properties["central_ini_cond"]

    # Opening JSON file
    with open("control_dicts/mesh_properties.json", "r") as openfile:
        # Reading from json file
        mesh_properties = json.load(openfile)

    h = mesh_properties["h"]
    k = mesh_properties["k"]
    x_dom = mesh_properties["x_dom"]
    y_dom = mesh_properties["y_dom"]
    t_dom = mesh_properties["t_dom"]

    Cl_list, Cp_list, speed_up_list = read_files("fvm_sim")

    Cp_fvm, Cl_fvm, center, radius = format_array(Cp_list[0], Cl_list[0])

    size_x, size_y, size_t = get_mesh_properties(x_dom, y_dom, t_dom, h, k)

    with open("source_points/lymph_vessels.pkl", "rb") as f:
        leu_source_points = pk.load(f)

    animate_1D_evolution(
        size_t,
        size_x,
        t_dom,
        x_dom,
        Cp_fvm,
        Cl_fvm,
        leu_source_points,
        5000,
        150,
        show=True,
    )
