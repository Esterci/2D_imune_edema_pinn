import torch
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Função original (convertida para Torch)
# ----------------------------
def apply_initial_conditions(ini_cond, Cb, cx, cy, radius, size_x, size_y):
    for i in range(size_x):
        for j in range(size_y):
            if (i - cx) ** 2 + (j - cy) ** 2 <= radius**2:
                Cb[i, j] = ini_cond
    return Cb

# ----------------------------
# Função fornecida pelo usuário
# ----------------------------
def generate_initial_points(num_points, device, center_x_tc, radius_tc, initial_tc):

    t = torch.zeros(num_points, 1, dtype=torch.float32)

    x = torch.rand(num_points, 1, dtype=torch.float32)

    euclidean_distances = ((x - center_x_tc.item()) ** 2) ** 0.5

    inside_circle_mask = euclidean_distances <= radius_tc.item()

    result = torch.cat([x, euclidean_distances, inside_circle_mask], dim=1)

    print(center_x_tc.item())
    print(radius_tc.item())
    print(result)

    C_init = torch.zeros((len(x), 2), dtype=torch.float32)

    C_init[:, 1] = inside_circle_mask.ravel() * initial_tc.ravel()

    return (
        (t.requires_grad_(True), x.requires_grad_(True)),
        C_init.to(device),
    )
# ----------------------------
# Definições básicas
# ----------------------------
device = "cpu"
size_x, size_y = 10, 1

# Campos base de concentração
Cb1 = torch.zeros((size_x, 1))
Cb2 = torch.zeros((size_x, 1))
Cb3 = torch.zeros((size_x, 1))
Cb4 = torch.zeros((size_x, 1))

# ----------------------------
# Aplica condições iniciais
# ----------------------------
Cb2 = apply_initial_conditions(1.0, Cb2, 5, 0, 2, size_x, size_y)

x = np.arange(0,1,0.1)

cx_real = 1/size_x * 5
r_real = 1/size_x * 2

# Gera os tensores de fronteira (contendo as últimas concentrações)
(_, x_tc), C = generate_initial_points(size_x*2, device, torch.tensor([cx_real]),torch.tensor([r_real]),torch.tensor([1]))

# Usa os valores das duas colunas de C como "concentrações" de referência
# (Aqui só para visualização: expandimos para uma malha 2D)
Cb3 = C[:, 0]  # apenas ilustrativo
Cb4 = C[:, 1]  # apenas ilustrativo

# ----------------------------
# Visualização
# ----------------------------
plt.figure(figsize=(8, 6))

configs = [
    (Cb1, "Cl fvm", "-o"),
    (Cb2, "Cp fvm", "-o"),
    (Cb3, "Cl pinn", "o"),
    (Cb4, "Cp pinn", "o")
]

for i,(Cb, title, line) in enumerate(configs):
    if i < 2:
        plt.plot(x, Cb, line, label=title, linewidth=2)
    else:
        plt.plot(x_tc.detach().numpy(), Cb, line, label=title, linewidth=2)

plt.xlabel("Posição (x)")
plt.ylabel("Concentração bacteriana (Cb)")
plt.title("Comparação das Quatro Condições Iniciais (Curvas)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

