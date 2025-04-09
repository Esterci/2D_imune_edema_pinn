import numpy as np
import pickle as pk
import time


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
