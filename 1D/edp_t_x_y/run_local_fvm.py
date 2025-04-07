import os

count = 0

for i in range(33):

    os.system("time ~/.conda/envs/torch-numba/bin/python3 fvm_comparison.py")
