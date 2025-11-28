import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pk
from glob import glob
import os
from math import ceil
from fisiocomPinn.Net import *
from fisiocomPinn.Trainer import *
from fisiocomPinn.Validator import *
from fisiocomPinn.Loss import *
from fisiocomPinn.Loss_PINN import *
from fisiocomPinn.Utils import *
import time
from utils import torch_random_indices

# Dictionary that maps string names to PyTorch activation classes.
# This lets you build networks from a compact string description.
activation_dict = {
    "Elu": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Softplus": nn.Softplus,
    "Tanh": nn.Tanh,
    "Linear": nn.Linear,
    "ReLU": nn.ReLU,
    "RReLU": nn.RReLU,
    "SELU": nn.SELU,
    "CELU": nn.CELU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "GLU": nn.GLU,
}


def generate_model(arch_str, input, output):
    """
    Builds a fully-connected neural network from a compact architecture string.

    Parameters
    ----------
    arch_str : str
        Architecture string, e.g. "Tanh--32__Tanh--32__Linear--16".
        Each block is "Activation--N_neurons" and blocks are separated by "__".
    input : int
        Number of input features (e.g. 2 for (t, x)).
    output : int
        Number of output features (e.g. 2 for (Cl, Cp)).

    Returns
    -------
    nn.Sequential
        A PyTorch sequential model with the specified architecture, in double precision.
    """
    hidden_layers = arch_str.split("__")

    modules = []

    for params in hidden_layers:
        if len(params) != 0:
            activation, out_neurons = params.split("--")

            # First layer: we connect from the specified input dimension
            if len(modules) == 0:
                if activation == "Linear":
                    # If the activation is "Linear", we create a Linear layer directly
                    modules.append(
                        activation_dict[activation](input, int(out_neurons)).double()
                    )

                else:
                    # Otherwise: Linear layer + non-linear activation
                    modules.append(nn.Linear(input, int(out_neurons)).double())
                    modules.append(activation_dict[activation]().double())

            else:
                # Subsequent layers: we connect from the previous hidden dimension
                if activation == "Linear":
                    modules.append(
                        activation_dict[activation](
                            int(in_neurons), int(out_neurons)
                        ).double()
                    )

                else:
                    modules.append(
                        nn.Linear(int(in_neurons), int(out_neurons)).double()
                    )
                    modules.append(activation_dict[activation]().double())

            # Update the input size for the next layer
            in_neurons = out_neurons

    # Final linear layer from last hidden layer to output dimension
    modules.append(nn.Linear(int(in_neurons), output).double())

    return nn.Sequential(*modules)


def get_infection_site(struct_name):
    """
    Extracts the infection center and radius from a filename encoded with metadata.

    The expected filename pattern is something like:
        'Cp__...__(x_center,y_center)__radius--R.pkl'
    where the last elements encode (center_x, center_y) and radius.

    Parameters
    ----------
    struct_name : str
        File path or file name that encodes the infection site.

    Returns
    -------
    center : tuple(float, float)
        Coordinates of the infection center.
    radius : float
        Infection radius.
    """
    center_str = (struct_name).split("__")[-3].split("(")[-1].split(")")[0].split(",")

    center = (float(center_str[0]), float(center_str[1]))

    radius = float(struct_name.split("__")[-2].split("--")[-1].split(".pkl")[0])

    return center, radius


def read_files(path):
    """
    Reads and separates pickled files containing Cp, Cl and speed_up information.

    The function expects files in 'path' whose names start with:
        'Cl__', 'Cp__' or 'speed_up__'.

    Parameters
    ----------
    path : str
        Directory path containing the pkl files.

    Returns
    -------
    Cl_list : list of str
        Sorted list of paths to Cl files.
    Cp_list : list of str
        Sorted list of paths to Cp files.
    speed_up_list : list of str
        Sorted list of paths to speed_up files.
    """
    file_list = sorted(glob(path + "/*"))

    speed_up_list = []
    Cl_list = []
    Cp_list = []

    for file in file_list:

        # Helper to extract the "variable" name from the file name (Cl, Cp, speed_up)
        variable = lambda a: a.split("/")[-1].split("__")[0]

        if variable(file) == "Cl":
            Cl_list.append(file)

        elif variable(file) == "Cp":
            Cp_list.append(file)

        elif variable(file) == "speed_up":
            speed_up_list.append(file)

    return Cl_list, Cp_list, speed_up_list


def format_array(Cp_file, Cl_file):
    """
    Loads Cp and Cl fields from pickle files and extracts infection site metadata.

    Parameters
    ----------
    Cp_file : str
        Path to the Cp solution file (pickled array).
    Cl_file : str
        Path to the Cl solution file (pickled array).

    Returns
    -------
    Cp : np.ndarray
        Pathogen concentration field.
    Cl : np.ndarray
        Leukocyte concentration field.
    center : tuple(float, float)
        Infection center coordinates extracted from file name.
    radius : float
        Infection radius extracted from file name.
    """
    with open(Cp_file, "rb") as f:
        Cp = pk.load(f)

    with open(Cl_file, "rb") as f:
        Cl = pk.load(f)

    center, radius = get_infection_site(Cp_file)

    return Cp, Cl, center, radius


def get_mesh_properties(
    x_dom,
    y_dom,
    t_dom,
    h,
    k,
    verbose=True,
):
    """
    Computes the number of grid points in space and time based on domain and step sizes.

    Parameters
    ----------
    x_dom, y_dom, t_dom : tuple(float, float)
        Spatial and temporal domain limits, e.g. (0, 1).
    h : float
        Spatial step size.
    k : float
        Time step size.
    verbose : bool
        Whether to print the mesh information.

    Returns
    -------
    (size_x, size_y, size_t) : tuple(int, int, int)
        Number of grid points in x, y and t.
    """

    size_x = int(((x_dom[1] - x_dom[0]) / (h)))
    size_y = int(((y_dom[1] - y_dom[0]) / (h)))
    size_t = int(((t_dom[1] - t_dom[0]) / (k)))

    if verbose:
        print(
            "Steps in time = {:d}\nSteps in space_x = {:d}\nSteps in space_y = {:d}\n".format(
                size_t,
                size_x,
                size_y,
            )
        )

    return (size_x, size_y, size_t)


def under_sampling(n_samples, Cl, Cp):
    """
    Performs temporal under-sampling of FVM solutions for Cl and Cp.

    The function selects 'n_samples' equally spaced time indices and
    extracts the corresponding snapshots from the full data.

    Parameters
    ----------
    n_samples : int
        Number of time snapshots to retain.
    Cl : np.ndarray
        Full leukocyte field with shape (n_t, n_x, n_y or n_x).
    Cp : np.ndarray
        Full pathogen field with shape (n_t, n_x, n_y or n_x).

    Returns
    -------
    reduced_Cl : np.ndarray
        Undersampled leukocyte field with shape (n_samples, ...).
    reduced_Cp : np.ndarray
        Undersampled pathogen field with shape (n_samples, ...).
    choosen_points : np.ndarray
        Indices of the selected time steps.
    """

    choosen_points = np.linspace(
        0, len(Cl) - 1, num=n_samples, endpoint=True, dtype=int
    )

    reduced_Cl = np.zeros((n_samples, Cl.shape[1], Cl.shape[2]))

    reduced_Cp = np.zeros((n_samples, Cp.shape[1], Cl.shape[2]))

    for i, idx in enumerate(choosen_points):

        reduced_Cl[i, :] = Cl[idx, :, :]

        reduced_Cp[i, :] = Cp[idx, :, :]

    return reduced_Cl, reduced_Cp, choosen_points


def create_input_mesh(
    source, t_dom, x_dom, size_t, size_x, n_samples=None, Cl_fvm=None, Cp_fvm=None
):
    """
    Creates the (t, x) input mesh and the corresponding source term for training or evaluation.

    If n_samples is provided, it also performs temporal undersampling of FVM solutions
    and returns the reduced meshes and FVM data.

    Parameters
    ----------
    source : np.ndarray
        Spatial distribution of source locations (e.g. leukocyte entry points) along x.
    t_dom : tuple(float, float)
        Time domain limits.
    x_dom : tuple(float, float)
        Spatial domain limits.
    size_t : int
        Number of time steps.
    size_x : int
        Number of spatial points.
    n_samples : int, optional
        Number of time snapshots to retain (for undersampling).
    Cl_fvm, Cp_fvm : np.ndarray, optional
        Full FVM solutions, required if undersampling is requested.

    Returns
    -------
    If n_samples is not None:
        reduced_Cl, reduced_Cp, t_mesh, x_mesh, source_mesh
    else:
        t_mesh, x_mesh, source_mesh
    """

    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float64
    )

    x_idx = np.linspace(0, size_x, num=size_x, endpoint=False, dtype=int)

    if n_samples:

        reduced_Cl, reduced_Cp, choosen_points = under_sampling(
            n_samples, Cl_fvm, Cp_fvm
        )

        t_np = np.linspace(
            t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float64
        )[choosen_points]

        # Generate mesh over selected times and all x indices
        x_idx_mesh, t_mesh = np.meshgrid(
            x_idx,
            t_np,
        )

        x_mesh = np.zeros_like(t_mesh)
        source_mesh = np.zeros_like(t_mesh)

        # Map indices to physical coordinates and source values
        x_mesh = x_np[x_idx_mesh.ravel()]
        source_mesh = source[x_idx_mesh.ravel()]

        return (
            reduced_Cl,
            reduced_Cp,
            t_mesh,
            x_mesh,
            source_mesh,
        )

    # Full (t, x) mesh without undersampling
    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float64)

    x_idx_mesh, t_mesh = np.meshgrid(
        x_idx,
        t_np,
    )

    x_mesh = np.zeros_like(t_mesh)
    source_mesh = np.zeros_like(t_mesh)

    x_mesh = x_np[x_idx_mesh.ravel()]
    source_mesh = source[x_idx_mesh.ravel()]

    return (
        t_mesh,
        x_mesh,
        source_mesh,
    )


def allocates_training_mesh(
    t_dom,
    x_dom,
    size_t,
    size_x,
    center_x,
    initial_cond,
    radius,
    Cp_fvm,
    Cl_fvm,
    source,
    n_samples=None,
):
    """
    Prepares and allocates all tensors required for training the PINN or standard NN.
    This includes:
    - the full (t, x) input mesh,
    - tensors encoding the initial condition, infection center, and radius,
    - the full FVM solution used as data-supervision targets,
    - and, optionally, an undersampled subset of the data for reduced training cost.

    Parameters
    ----------
    t_dom, x_dom : tuple(float, float)
        Time and space domain limits (e.g., (0, 1)).
    size_t, size_x : int
        Number of discretization points in time and space.
    center_x : float
        x–coordinate of the infection center extracted from metadata.
    initial_cond : float or array-like
        Initial condition values for (Cl, Cp).
    radius : float
        Infection radius extracted from metadata.
    Cp_fvm, Cl_fvm : np.ndarray
        Finite Volume Method (FVM) solutions over the full space–time grid.
    source : np.ndarray
        Spatial distribution of source activation along the spatial domain.
    n_samples : int, optional
        Number of temporal samples to retain if undersampling is desired.

    Returns
    -------
    Without undersampling (n_samples=None):
        initial_tc, center_x_tc, radius_tc,
        t_tc, x_tc, src_tc, target, device

    With undersampling (n_samples provided):
        initial_tc, center_x_tc, radius_tc,
        t_tc, x_tc, src_tc, target,
        reduced_t_tc, reduced_x_tc,
        reduced_src_tc, reduced_target,
        device
    """

    # Full tensorized (t, x) mesh and associated source mask
    (
        t_mesh,
        x_mesh,
        src_mesh,
    ) = create_input_mesh(source, t_dom, x_dom, size_t, size_x)

    # Select computational device (GPU preferred, CPU fallback)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    print("device:", device)

    # Convert metadata scalars to double-precision tensors
    # These can have gradients if required during training
    initial_tc = (
        torch.tensor(initial_cond, dtype=torch.float64)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    center_x_tc = (
        torch.tensor(center_x, dtype=torch.float64)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    radius_tc = (
        torch.tensor(radius, dtype=torch.float64)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    # Convert full (t, x) grid into training tensors
    # Each position is flattened into a column vector
    t_tc = torch.tensor(t_mesh, dtype=torch.float64).reshape(-1, 1).requires_grad_(True).to(device)
    x_tc = torch.tensor(x_mesh, dtype=torch.float64).reshape(-1, 1).requires_grad_(True).to(device)

    # Source mask over the full mesh
    src_tc = (
        torch.tensor(src_mesh, dtype=torch.float64)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    # FVM solution flattened such that:
    #   target[i] = [Cl_fvm, Cp_fvm] at the same (t,x) point as t_tc[i], x_tc[i]
    target = torch.tensor(
        np.array([Cl_fvm.flatten(), Cp_fvm.flatten()]).T,
        dtype=torch.float64,
    )

    # If undersampling is enabled, repeat the mesh generation procedure
    # but restricted to n_samples selected time slices
    if n_samples:

        (
            reduced_Cl,
            reduced_Cp,
            reduced_t_mesh,
            reduced_x_mesh,
            reduced_src_mesh,
        ) = create_input_mesh(
            source,
            t_dom,
            x_dom,
            size_t,
            size_x,
            n_samples,
            Cl_fvm,
            Cp_fvm,
        )

        # Undersampled temporal and spatial coordinates
        reduced_t_tc = torch.tensor(reduced_t_mesh, dtype=torch.float64).reshape(-1, 1).requires_grad_(True).to(device)
        reduced_x_tc = torch.tensor(reduced_x_mesh, dtype=torch.float64).reshape(-1, 1).requires_grad_(True).to(device)

        # Undersampled source mask
        reduced_src_tc = (
            torch.tensor(reduced_src_mesh, dtype=torch.float64)
            .reshape(-1, 1)
            .requires_grad_(True)
        )

        # Undersampled FVM targets, aligned with the reduced mesh
        reduced_target = torch.tensor(
            np.array([reduced_Cl.flatten(), reduced_Cp.flatten()]).T,
            dtype=torch.float64,
        )

        return (
            initial_tc,
            center_x_tc,
            radius_tc,
            t_tc,
            x_tc,
            src_tc,
            target,
            reduced_t_tc,
            reduced_x_tc,
            reduced_src_tc,
            reduced_target,
            device,
        )

    # Return full dataset without undersampling
    else:
        return (
            initial_tc,
            center_x_tc,
            radius_tc,
            t_tc,
            x_tc,
            src_tc,
            target,
            device,
        )


def generate_initial_points(num_points, device, size_x):
    """
    Generates random initial condition points at t = 0 along the spatial domain [0, 1).

    The initial condition is defined as a superposition of two Gaussian peaks:
    one near x = 0 (for Cl) and one near x = 1 - 1/size_x (for Cp).

    Parameters
    ----------
    num_points : int
        Number of initial condition points.
    device : torch.device
        Device where tensors will be allocated.
    size_x : int
        Number of spatial grid points (used to locate the second Gaussian peak).

    Returns
    -------
    (t, x) : tuple(torch.Tensor, torch.Tensor)
        Tensors of shape (num_points, 1) for time and space, with requires_grad=True.
    C_init : torch.Tensor
        Initial condition values of shape (num_points, 2) for (Cl, Cp).
    """

    t = torch.zeros(num_points, 1, dtype=torch.float64)

    x = torch.rand(num_points, 1, dtype=torch.float64)

    # Parameters of the two Gaussian peaks
    a = 0
    a2 = 1 - (1 / size_x)
    b = 4
    c = 2

    C_init = torch.zeros((len(x), 2), dtype=torch.float64)

    # Gaussian centered at a (near x = 0) for Cl
    C_init[:, 0] = torch.exp(-(((x.reshape(1, -1) - a) * b) ** 2)) / c

    # Gaussian centered at a2 (near x = 1 - 1/size_x) for Cp
    C_init[:, 1] = torch.exp(-(((x.reshape(1, -1) - a2) * b) ** 2)) / c

    return (
        (t.requires_grad_(True), x.requires_grad_(True)),
        C_init.to(device),
    )


def initial_condition(batch, model, device):
    """
    Evaluates the model at initial condition points.

    Parameters
    ----------
    batch : tuple(torch.Tensor, torch.Tensor)
        (t, x) tensors for t = 0.
    model : nn.Module
        Neural network model (PINN or standard NN).
    device : torch.device
        Device to evaluate the model.

    Returns
    -------
    torch.Tensor
        Model predictions at the initial points.
    """
    t, x = batch

    input_data = torch.cat([t, x], dim=1).to(device)

    return model(input_data)


def generate_boundary_points(num_points, device, t_upper):
    """
    Generates random boundary points at x = 0 and x = 1 for 0 <= t <= t_upper.

    Parameters
    ----------
    num_points : int
        Total number of boundary points (split equally at left and right boundaries).
    device : torch.device
        Device for the tensors.
    t_upper : float
        Upper bound of the time interval.

    Returns
    -------
    (t, x) : tuple(torch.Tensor, torch.Tensor)
        Boundary points (t, x), with x in {0, 1}.
    C : torch.Tensor
        Zero boundary values for (Cl, Cp), assumed here as homogeneous boundary condition.
    """

    t = torch.rand(num_points, 1, dtype=torch.float64) * t_upper

    # Alternate x between 0 and 1 along the batch
    x = (
        torch.tensor([0.0, 1], dtype=torch.float64)
        .repeat(num_points // 2, 1)
        .view(-1, 1)
    )

    C = torch.zeros((len(x), 2), dtype=torch.float64)

    return (
        (t.requires_grad_(True), x.requires_grad_(True)),
        C.to(device),
    )


def boundary_condition(batch, model, Dn, X_nb, Db, device):
    """
    Computes the boundary residual for the PDE system.

    Here, we compute:
        - Dn * dCl/dx * n  (Neumann-like flux for Cl)
        - Db * dCp/dx      (Neumann-like flux for Cp, with unit normal)
    at x = 0 and x = 1.

    Parameters
    ----------
    batch : tuple(torch.Tensor, torch.Tensor)
        (t, x) boundary points.
    model : nn.Module
        Neural network approximating (Cl, Cp).
    Dn : float
        Diffusion coefficient for leukocytes.
    X_nb : float
        Chemotaxis coefficient (not explicitly used here but available).
    Db : float
        Diffusion coefficient for pathogens.
    device : torch.device
        Device for computations.

    Returns
    -------
    torch.Tensor
        Boundary residuals concatenated as [Cl_boundary, Cp_boundary] with shape (N, 2).
    """

    t, x = batch

    input_data = torch.cat([t, x], dim=1).to(device)

    pred = model(input_data)

    # Outward normal at x = 0 is -1 and at x = 1 is +1.
    # We alternate these entries along the batch.
    n = (
        torch.tensor([-1, 1], dtype=torch.float64)
        .repeat(len(pred) // 2, 1)
        .requires_grad_(True)
        .view(-1, 1)
        .to(device)
    )

    # First spatial derivative for Cl
    dCl_dx = torch.autograd.grad(
        pred[:, 0],
        x,
        torch.ones_like(pred[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    # First spatial derivative for Cp
    dCp_dx = torch.autograd.grad(
        pred[:, 1],
        x,
        torch.ones_like(pred[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    # Flux-type boundary condition for Cl (depends on Dn and normal direction)
    Cl_boundary = Dn * dCl_dx * n

    # Flux-type boundary condition for Cp (here normal is implicitly +1)
    Cp_boundary = Db * dCp_dx

    # Return concatenated boundary residuals
    return torch.cat([Cl_boundary, Cp_boundary], dim=1)


def generate_pde_points(num_points, device, t_upper):
    """
    Generates random interior collocation points (t, x) for enforcing the PDE residual.

    Parameters
    ----------
    num_points : int
        Number of PDE collocation points.
    device : torch.device
        Device for the tensors.
    t_upper : float
        Upper bound for time sampling in [0, t_upper].

    Returns
    -------
    (t, x) : tuple(torch.Tensor, torch.Tensor)
        PDE collocation points with requires_grad=True.
    C : torch.Tensor
        Zero tensor of shape (num_points, 2) used as a placeholder or target.
    """
    # Random time in [0, t_upper]
    t = torch.rand(num_points, 1, dtype=torch.float64) * t_upper

    # Random spatial coordinate in [0, 1)
    x = torch.rand(num_points, 1, dtype=torch.float64)

    C = torch.zeros((len(x), 2), dtype=torch.float64)

    return (
        (t.requires_grad_(True), x.requires_grad_(True)),
        C.to(device),
    )


def sort_pde_points(
        t_tc,
            x_tc,
            src_tc,
            num_points, device, t_upper):

    sort_idx = idx_torch_random_indices(t_tc, num_points)
    
    # Random time in [0, t_upper]
    sorted_t = t_tc[sort_idx]

    sorted_x = x_tc[sort_idx]
    
    sorted_src = src_tc[sort_idx]
    
    C = torch.zeros((len(x), 2), dtype=torch.float64)

    return (
        (sorted_t.requires_grad_(True), sorted_x.requires_grad_(True), sorted_src.requires_grad_(True)),
        C.to(device),
    )


def generate_pde_source(original_source, h, batch, device):
    """
    Generates a source term for PDE collocation points based on the original_source field.

    For each x in the batch, it checks whether it falls within a distance h
    from any active source in original_source, and marks those points as 1.0.

    Parameters
    ----------
    original_source : torch.Tensor
        1D tensor over the spatial grid indicating source locations (0 or 1).
    h : float
        Spatial step size used to define the neighborhood around each source.
    batch : torch.Tensor
        Batch of PDE collocation points (t, x) or only x if already split.
    device : torch.device
        Device for computations.

    Returns
    -------
    new_source : torch.Tensor
        1D tensor of shape (B, 1) with 1.0 for points near a source and 0 otherwise.
    """

    _, x_batch = batch.tensor_split(2, dim=1)  # [B, 1]
    x_domain = torch.arange(0, 1, h).view(-1, 1).to(device)  # [N, 1]

    # Identify active source locations in x_domain
    source_locs = x_domain[original_source.view(-1) == 1]  # shape [M, 1], where M ≤ N
    source_locs

    # Compute bounds (one cell to the left and right of each active source)
    l_bound = source_locs - h  # [M, 1]
    u_bound = source_locs + h  # [M, 1]

    # Broadcast and check
    x_batch_exp = x_batch[:, None, :]  # [B, 1, 1]
    l_bound_exp = l_bound[None, :, :]  # [1, M, 1]
    u_bound_exp = u_bound[None, :, :]  # [1, M, 1]

    # Check if x_batch[i] is within any [l_bound[j], u_bound[j]]
    in_range = (x_batch_exp > l_bound_exp) & (x_batch_exp < u_bound_exp)  # [B, M, 1]
    match = in_range.any(dim=1)  # [B, 1]

    # Generate new source mask for PDE points
    new_source = torch.zeros_like(x_batch)
    new_source[match] = 1.0

    return new_source

def pde(
    batch,
    model,
    h,
    cb,
    phi,
    lambd_nb,
    Db,
    y_n,
    Cn_max,
    lambd_bn,
    mi_n,
    Dn,
    X_nb,
    device,
):
    """
    Computes the PDE residuals for the coupled system (Cl, Cp) at collocation points.

    The model predicts Cl and Cp; then the function computes temporal and spatial
    derivatives via autograd and constructs the residuals based on the governing equations.

    Parameters
    ----------
    batch : tuple(torch.Tensor, torch.Tensor)
        (t, x) collocation points for PDE.
    model : nn.Module
        Neural network approximating (Cl, Cp).
    h : float
        Spatial step (currently not explicitly used here).
    cb : float
        Pathogen reproduction rate (for qb = cb * Cp).
    phi : float
        Porosity (ϕ_f) multiplying time derivatives.
    lambd_nb : float
        Phagocytosis rate λ_nb (for rb term).
    Db : float
        Diffusion coefficient for pathogens.
    y_n : float
        Permeability γ_n for leukocytes (for qn term).
    Cn_max : float
        Maximum leukocyte concentration (in blood).
    lambd_bn : float
        Leukocyte death rate induced by phagocytosis λ_bn.
    mi_n : float
        Natural leukocyte death rate μ_n.
    Dn : float
        Leukocyte diffusion coefficient.
    X_nb : float
        Chemotaxis coefficient χ_nb.
    device : torch.device
        Device for computations.

    Returns
    -------
    torch.Tensor
        PDE residuals concatenated as [Cl_eq, Cp_eq] with shape (N, 2).
    """

    t, x = batch

    input_data = torch.cat([t, x], dim=1).to(device)

    # Network predictions: pred[:, 0] = Cl, pred[:, 1] = Cp
    pred = model(input_data)

    # First spatial derivatives
    dCl_dx = torch.autograd.grad(
        pred[:, 0],
        x,
        torch.ones_like(pred[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    dCp_dx = torch.autograd.grad(
        pred[:, 1],
        x,
        torch.ones_like(pred[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    # Time derivatives
    dCl_dt = torch.autograd.grad(
        pred[:, 0],
        t,
        torch.ones_like(pred[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    dCp_dt = torch.autograd.grad(
        pred[:, 1],
        t,
        torch.ones_like(pred[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    # Second spatial derivatives
    d2Cl_dx2 = torch.autograd.grad(
        dCl_dx,
        x,
        torch.ones_like(dCl_dx),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    d2Cp_dx2 = torch.autograd.grad(
        dCp_dx,
        x,
        torch.ones_like(dCp_dx),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    # Leukocyte source term qn = γ_n * Cp * (Cn_max - Cl)
    qn = y_n * pred[:, 1].ravel() * (Cn_max - pred[:, 0])  # [N,]

    # Leukocyte removal term rn = λ_bn * Cl * Cp + μ_n * Cl
    rn = lambd_bn * pred[:, 0].ravel() * pred[:, 1] + mi_n * pred[:, 0]  # [N,]

    # PDE residual for Cl (here only chemotaxis and time term are explicitly included)
    # Cl_eq = φ * ∂Cl/∂t - Dn ∂²Cl/∂x² - χ_nb (Cl ∂²Cp/∂x² + ∂Cl/∂x ∂Cp/∂x) - (qn - rn)
    # In this implementation you are using only part of this structure (chemotaxis + time)
    Cl_eq = (
        -X_nb * ((dCl_dx * dCp_dx).ravel() + pred[:, 0] * d2Cp_dx2.ravel())
        - dCl_dt.ravel() * phi
    )

    # Pathogen source and sink terms
    qb = cb * pred[:, 1]

    rb = lambd_nb * pred[:, 0].ravel() * pred[:, 1]

    # PDE residual for Cp (currently only time term * porosity is included)
    # The full equation would typically be:
    #   φ * ∂Cp/∂t - Db ∂²Cp/∂x² - (qb - rb) = 0
    Cp_eq = -dCp_dt.ravel() * phi

    # Return residuals as a [N, 2] tensor
    return torch.cat([Cl_eq.reshape(-1, 1), Cp_eq.reshape(-1, 1)], dim=1)
