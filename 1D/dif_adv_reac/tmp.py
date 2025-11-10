import matplotlib.pyplot as plt
import torch

N = 100
x = torch.linspace(0, 0.99, N)

a = 0
b = 4
c = 2

y = torch.exp(-(((x - a) * b) ** 2)) / c

a2 = 0.99

y2 = torch.exp(-(((x - a2) * b) ** 2)) / c

plt.plot(x, y, label=f"p=2")
plt.plot(x, y2, label=f"p=2")

plt.legend()
plt.show()
