import pickle as pk
import json
import os

import torch
import numpy as np

from pinn import *
from utils import *
from sklearn.model_selection import KFold
from grid_search import normalize_target

arch_str = "Tanh--32__Tanh--32__Tanh--32__Tanh--32__Tanh--32"

beta1 = 0.9
beta2 = 0.9999

k_folds = 5

# Estes valores são frações do desvio padrão do sinal:
# 0.01 = 1%, 0.03 = 3%, 0.05 = 5%, 0.10 = 10%
noise_levels = [0, 0.01, 0.03, 0.05, 0.10]

samples_levels = [50, 30, 20, 1]


def add_white_gaussian_noise(y, noise_level=0.05, clamp_min=0.0):
    """
    Adiciona ruído gaussiano branco proporcional ao desvio padrão de cada variável.

    y: tensor com shape (N, 2), por exemplo [Cl, Cp]
    noise_level: fração do desvio padrão.
                 Ex: 0.05 = ruído com sigma igual a 5% do std de cada variável.
    clamp_min: se não for None, limita os valores inferiores. Útil para evitar
               concentrações negativas.
    """

    sigma = noise_level * torch.std(y, dim=0, keepdim=True)
    noise = sigma * torch.randn_like(y)

    y_noisy = y + noise

    if clamp_min is not None:
        y_noisy = torch.clamp(y_noisy, min=clamp_min)

    return y_noisy


def calculate_taylor_metrics(pred, ref, variable_names=("Cl", "Cp")):
    """
    Calcula métricas para diagrama de Taylor.

    pred: tensor ou array com shape (N, 2)
    ref:  tensor ou array com shape (N, 2)

    Para cada variável, retorna:
    - std_ref
    - std_pred
    - correlation
    - centered_rmse
    - normalized_std
    - normalized_centered_rmse
    - rmse
    - bias
    """

    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()

    if torch.is_tensor(ref):
        ref = ref.detach().cpu().numpy()

    metrics = {}

    for j, name in enumerate(variable_names):
        y_pred = pred[:, j].ravel()
        y_ref = ref[:, j].ravel()

        mean_pred = np.mean(y_pred)
        mean_ref = np.mean(y_ref)

        std_pred = np.std(y_pred, ddof=1)
        std_ref = np.std(y_ref, ddof=1)

        if std_pred == 0 or std_ref == 0:
            correlation = np.nan
        else:
            correlation = np.corrcoef(y_ref, y_pred)[0, 1]

        centered_rmse = np.sqrt(
            np.mean(((y_pred - mean_pred) - (y_ref - mean_ref)) ** 2)
        )

        rmse = np.sqrt(np.mean((y_pred - y_ref) ** 2))
        bias = mean_pred - mean_ref

        if std_ref == 0:
            normalized_std = np.nan
            normalized_centered_rmse = np.nan
        else:
            normalized_std = std_pred / std_ref
            normalized_centered_rmse = centered_rmse / std_ref

        metrics[name] = {
            "std_ref": float(std_ref),
            "std_pred": float(std_pred),
            "correlation": float(correlation),
            "centered_rmse": float(centered_rmse),
            "normalized_std": float(normalized_std),
            "normalized_centered_rmse": float(normalized_centered_rmse),
            "rmse": float(rmse),
            "bias": float(bias),
        }

    return metrics


def predict_model(model, data, device):
    """
    Faz inferência sem gradiente e retorna a predição em CPU.
    """

    model.eval()

    with torch.no_grad():
        pred = model(data.to(device))

    return pred.detach().cpu()


def main():

    os.makedirs("nn_sim", exist_ok=True)

    # =====================================================
    # Propriedades constantes
    # =====================================================

    with open("control_dicts/constant_properties.json", "r") as openfile:
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

    # =====================================================
    # Propriedades da malha
    # =====================================================

    with open("control_dicts/mesh_properties.json", "r") as openfile:
        mesh_properties = json.load(openfile)

    h = mesh_properties["h"]
    k = mesh_properties["k"]
    x_dom = mesh_properties["x_dom"]
    y_dom = mesh_properties["y_dom"]
    t_dom = mesh_properties["t_dom"]

    # =====================================================
    # Dados do MVF
    # =====================================================

    Cl_list, Cp_list, speed_up_list = read_files("fvm_sim")

    Cp_fvm, Cl_fvm, center, radius = format_array(Cp_list[0], Cl_list[0])

    size_x, size_y, size_t = get_mesh_properties(x_dom, y_dom, t_dom, h, k)

    print("center:", center)
    print("radius:", radius)
    print("central_ini_cond:", central_ini_cond)

    # =====================================================
    # Experimentos
    # =====================================================

    for percent in samples_levels:
        for noise_level in noise_levels:

            print("\n" + "#" * 60)
            print(f"Samples percent: {percent}")
            print(f"Noise level: {noise_level}")
            print("#" * 60)

            (
                initial_tc,
                center_x_tc,
                radius_tc,
                data_tc,
                target,
                data_tc,
                target,
                device,
            ) = allocates_training_mesh(
                t_dom,
                x_dom,
                size_t,
                size_x,
                center[0],
                central_ini_cond,
                radius,
                Cl_fvm,
                Cp_fvm,
                samples_percent=percent,
            )

            n_epochs = int(1e4)
            pinn_batch = int(1e4)

            kfold = KFold(
                n_splits=k_folds,
                shuffle=True,
                random_state=42,
            )

            # =====================================================
            # Acumuladores out-of-fold
            # =====================================================

            pinn_pred_all = torch.empty_like(target.detach().cpu())
            nn_pred_all = torch.empty_like(target.detach().cpu())

            target_all = target.detach().cpu().clone()

            min_cl, min_cp, delta_cl, delta_cp, target_norm = normalize_target(target)

            pinn_fold_mae = []
            nn_fold_mae = []

            pinn_fold_rmse = []
            nn_fold_rmse = []

            pinn_train_time = []
            nn_train_time = []

            pinn_min_mae = np.inf
            nn_min_mae = np.inf

            pinn_best_pred = None
            nn_best_pred = None

            pinn_best_target = None
            nn_best_target = None

            # =====================================================
            # K-fold
            # =====================================================

            for fold, (train_idx, test_idx) in enumerate(kfold.split(data_tc)):

                print("\n" + "=" * 30)
                print(f"Fold {fold + 1}/{k_folds}")
                print("=" * 30)

                test_idx_torch = torch.as_tensor(test_idx, dtype=torch.long)

                with torch.no_grad():
                    data_train = data_tc[train_idx]
                    target_train_clean = target_norm[train_idx]

                    target_train = add_white_gaussian_noise(
                        target_train_clean,
                        noise_level=noise_level,
                        clamp_min=0.0,
                    )

                    data_test = data_tc[test_idx]
                    target_test = target_norm[test_idx]
                    target_test_cpu = target_test.detach().cpu()

                batch_size = max(int(len(data_train) / 10), 1)

                # =====================================================
                # PINN
                # =====================================================

                pinn_model = generate_model(arch_str, 2, 2).to(device)

                print("\nPINN")
                print(
                    "Number of parameters:",
                    sum(p.numel() for p in pinn_model.parameters() if p.requires_grad),
                )

                pinn_model, _, train_time = pinn_training(
                    n_epochs,
                    batch_size,
                    pinn_model,
                    device,
                    beta1,
                    beta2,
                    pinn_batch,
                    center_x_tc,
                    radius_tc,
                    initial_tc,
                    t_dom,
                    Dn,
                    X_nb,
                    Db,
                    cb,
                    phi,
                    lambd_nb,
                    y_n,
                    Cn_max,
                    lambd_bn,
                    mi_n,
                    data_train,
                    target_train,
                    delta_cl,
                    delta_cp,
                    min_cl,
                    min_cp,
                )

                pinn_pred = predict_model(
                    pinn_model,
                    data_test,
                    device,
                )

                pinn_pred_all[test_idx_torch] = pinn_pred
                target_all[test_idx_torch] = target_test_cpu

                pinn_abs_error = torch.abs(pinn_pred - target_test_cpu)
                pinn_mae = torch.mean(pinn_abs_error).item()
                pinn_rmse = torch.sqrt(
                    torch.mean((pinn_pred - target_test_cpu) ** 2)
                ).item()

                pinn_fold_mae.append(pinn_mae)
                pinn_fold_rmse.append(pinn_rmse)
                pinn_train_time.append(train_time)

                if pinn_mae <= pinn_min_mae:
                    pinn_min_mae = pinn_mae
                    pinn_best_pred = pinn_pred.clone()
                    pinn_best_target = target_test_cpu.clone()

                # =====================================================
                # NN
                # =====================================================

                nn_model = generate_model(arch_str, 2, 2).to(device)

                print("\nNN")
                print(
                    "Number of parameters:",
                    sum(p.numel() for p in nn_model.parameters() if p.requires_grad),
                )

                nn_model, _, train_time = nn_training(
                    n_epochs,
                    batch_size,
                    nn_model,
                    device,
                    beta1,
                    beta2,
                    data_train,
                    target_train,
                )

                nn_pred = predict_model(
                    nn_model,
                    data_test,
                    device,
                )

                nn_pred_all[test_idx_torch] = nn_pred

                nn_abs_error = torch.abs(nn_pred - target_test_cpu)
                nn_mae = torch.mean(nn_abs_error).item()
                nn_rmse = torch.sqrt(
                    torch.mean((nn_pred - target_test_cpu) ** 2)
                ).item()

                nn_fold_mae.append(nn_mae)
                nn_fold_rmse.append(nn_rmse)
                nn_train_time.append(train_time)

                if nn_mae <= nn_min_mae:
                    nn_min_mae = nn_mae
                    nn_best_pred = nn_pred.clone()
                    nn_best_target = target_test_cpu.clone()

                del pinn_model
                del nn_model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # =====================================================
            # Métricas Taylor após todos os folds
            # =====================================================

            pinn_taylor_metrics = calculate_taylor_metrics(
                pinn_pred_all,
                target_all,
                variable_names=("Cl", "Cp"),
            )

            nn_taylor_metrics = calculate_taylor_metrics(
                nn_pred_all,
                target_all,
                variable_names=("Cl", "Cp"),
            )

            # =====================================================
            # Saídas
            # =====================================================

            suffix = f"noise--{noise_level}__pct--{percent}"

            with open(
                f"experiments/pinn__prediction_all__{suffix}.pkl", "wb"
            ) as openfile:
                pk.dump(
                    {
                        "pred": pinn_pred_all,
                        "target": target_all,
                        "taylor_metrics": pinn_taylor_metrics,
                    },
                    openfile,
                )

            with open(
                f"experiments/nn__prediction_all__{suffix}.pkl", "wb"
            ) as openfile:
                pk.dump(
                    {
                        "pred": nn_pred_all,
                        "target": target_all,
                        "taylor_metrics": nn_taylor_metrics,
                    },
                    openfile,
                )

            with open(f"experiments/pinn__best_fold__{suffix}.pkl", "wb") as openfile:
                pk.dump(
                    {
                        "pred": pinn_best_pred,
                        "target": pinn_best_target,
                        "mae": pinn_min_mae,
                    },
                    openfile,
                )

            with open(f"experiments/nn__best_fold__{suffix}.pkl", "wb") as openfile:
                pk.dump(
                    {
                        "pred": nn_best_pred,
                        "target": nn_best_target,
                        "mae": nn_min_mae,
                    },
                    openfile,
                )

            print("\nTaylor metrics - PINN")
            print(pinn_taylor_metrics)

            print("\nTaylor metrics - NN")
            print(nn_taylor_metrics)


if __name__ == "__main__":
    main()
