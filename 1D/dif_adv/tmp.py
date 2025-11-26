import matplotlib.pyplot as plt
import torch


#input_data.shape: torch.Size([20000, 2])
#pred.shape: torch.Size([20000, 2])
#dCl_dx.shape: torch.Size([20000, 1])
#dCp_dx.shape: torch.Size([20000, 1])
#dCl_dt.shape: torch.Size([20000, 1])
#dCp_dt.shape: torch.Size([20000, 1])
#d2Cl_dx2.shape: torch.Size([20000, 1])
#d2Cp_dx2.shape: torch.Size([20000, 1])


N = 100
size_x = 40
x = torch.rand(N, 1, dtype=torch.float64)

a = 0
a2 = 1 - (1 / size_x)
b = 4
c = 2

y = torch.exp(-(((x - a) * b) ** 2)) / c
y2 = torch.exp(-(((x - a2) * b) ** 2)) / c

plt.scatter(x, y, label=f"p=2")
plt.scatter(x, y2, label=f"p=2")

plt.legend()
plt.show()
