import matplotlib.pyplot as plt
import torch

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
