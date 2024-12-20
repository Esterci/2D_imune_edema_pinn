from fvm_model_serial import solve_pde
from fvm_model_parallel import cu_solve_pde
import numpy as np
import time
import pickle as pk
import json
import math
from utils import init_mesh
from numba import cuda

# List to store speed-up factors for each iteration
speed_up_list = []

# Load constant properties from JSON file
with open("control_dicts/constant_properties.json", "r") as openfile:
    constant_properties = json.load(openfile)

# Load mesh properties from JSON file
with open("control_dicts/mesh_properties.json", "r") as openfile:
    mesh_properties = json.load(openfile)


# Extracting constants and parameters
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
ini_cond_var = constant_properties["ini_cond_var"]

h = mesh_properties["h"]
k = mesh_properties["k"]
x_dom = mesh_properties["x_dom"]
y_dom = mesh_properties["y_dom"]
t_dom = mesh_properties["t_dom"]

# Generate random initial condition parameters
center = (np.random.rand(), np.random.rand())
radius = np.random.rand() * (0.2 - 0.1) + 0.1

# Initialize mesh and related properties
size_x, size_y, size_t, initial_cond, leu_source_points, struct_name = init_mesh(
    x_dom,
    y_dom,
    t_dom,
    h,
    k,
    center,
    radius,
    central_ini_cond,
    ini_cond_var,
    1,
    create_source=False,
    source_type="central",
)

print(f"Mesh initialized for iteration.")

start = time.time()

# Initialize concentration arrays for neutrophils (Cn) and bacteria (Cb)
Cb_init = np.zeros((len(initial_cond), size_t, size_x, size_y))
Cn_init = np.zeros((len(initial_cond), size_t, size_x, size_y))

# Solve PDE for each initial condition in serial mode
for ini_index in range(len(initial_cond)):

    Cb, Cn = solve_pde(
        leu_source_points,
        size_t,
        size_x,
        size_y,
        h,
        k,
        Db,
        Dn,
        phi,
        cb,
        lambd_nb,
        mi_n,
        lambd_bn,
        y_n,
        Cn_max,
        X_nb,
        initial_cond[ini_index],
        center=center,
        radius=radius,
        verbose=False,
    )

    Cb_init[ini_index, :, :] = Cb
    Cn_init[ini_index, :, :] = Cn

end = time.time()

serial_time = end - start

print(f"Serial computation time for iteration: {serial_time:.2f} seconds.")

# Save results of serial computation
with open(f"fvm_sim/Cp__{struct_name}.pkl", "wb") as f:
    pk.dump(Cb_init, f)

with open(f"fvm_sim/Cl__{struct_name}.pkl", "wb") as f:
    pk.dump(Cn_init, f)

# Define CUDA threads and blocks
threadsperblock = (size_x, size_y)
blockspergrid_x = math.ceil(size_x / threadsperblock[0])
blockspergrid_y = math.ceil(size_y / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

start = time.time()

# Solve PDE using CUDA
for ini_cond in initial_cond:

    # Initialize device arrays for concentrations and sources
    Cb_buf_0 = cuda.to_device(np.zeros((size_x, size_y)))
    Cn_buf_0 = cuda.to_device(np.zeros((size_x, size_y)))
    device_leu_source = cuda.to_device(leu_source_points)

    # Additional buffers for synchronization
    Cb_buf_1 = cuda.device_array_like(Cb_buf_0)
    Cn_buf_1 = cuda.device_array_like(Cn_buf_0)

    # Arrays to store results for each time step
    Cb_final_device = cuda.to_device(np.zeros((size_t, size_x, size_y)))
    Cn_final_device = cuda.to_device(np.zeros((size_t, size_x, size_y)))

    cu_solve_pde[threadsperblock, blockspergrid](
        Cb_buf_0,
        Cn_buf_0,
        Cb_buf_1,
        Cn_buf_1,
        Cb_final_device,
        Cn_final_device,
        device_leu_source,
        size_t,
        size_x,
        size_y,
        h,
        k,
        Db,
        Dn,
        phi,
        cb,
        lambd_nb,
        mi_n,
        lambd_bn,
        y_n,
        Cn_max,
        X_nb,
        ini_cond,
        center,
        radius,
    )

    # Copy results back to the host
    Cb_host = np.empty(shape=Cb_final_device.shape, dtype=Cb_final_device.dtype)
    Cb_final_device.copy_to_host(Cb_host)

    Cn_host = np.empty(shape=Cn_final_device.shape, dtype=Cn_final_device.dtype)
    Cn_final_device.copy_to_host(Cn_host)

end = time.time()

cuda_time = end - start
print(f"CUDA computation time for iteration: {cuda_time:.2f} seconds.")

# Compute speed-up factor and store it
speed_up = serial_time / cuda_time
print(f"Speed-up for iteration: {speed_up:.2f}x")

# Save speed-up
with open("fvm_sim/speed_up__" + struct_name + ".pkl", "wb") as f:
    pk.dump(speed_up, f)
