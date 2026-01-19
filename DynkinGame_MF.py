import torch
import numpy as np
import os
import json
import csv
import time

from networkx.utils.decorators import np_random_state

from DRBSDE import fbsde
from DRBSDE import BSDEiter
from DRBSDE import Model
from DRBSDE import Result

path_base = "state_dicts/"

new_folder_flag = True
new_folder = "MeanField_4/"

folder_explicit = os.path.join(new_folder, "Explicit/")
folder_empirical = os.path.join(new_folder, "Empirical/")

for folder in [folder_explicit, folder_empirical]:
    os.makedirs(folder, exist_ok=True)

graph_path = os.path.join(new_folder, "Graphs/")
os.makedirs(graph_path, exist_ok=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mode = "Training"
mode = "Testing"
#ht_analysis = True
ht_analysis=False

def b(t, x):
    # x: shape [batch_size, dim_x]
    kappa_t = torch.tensor(kappa, dtype=torch.float32, device=device)
    beta_t = torch.tensor(beta, dtype=torch.float32, device=device)
    x0_t = torch.tensor(x0_value, device=device)
    # deterministic mean m_t
    m_t = x0_t * torch.exp((-kappa_t + beta_t) * t)
    drift = -kappa_t * x + beta_t * m_t
    return drift

def b_empirical(t, X):
    # mean-field drift (McKean-Vlasov)
    kappa_t = torch.tensor(kappa, dtype=torch.float32, device=device)
    beta_t = torch.tensor(beta, dtype=torch.float32, device=device)
    # empirical mean over batch
    m_t = X.mean(dim=0)
    drift = -kappa_t * X + beta_t * m_t
    return drift

def sigma(t, x):
    # x: shape [batch_size, dim_x, dim_d]
    sig_t = torch.tensor(sig, device = device)
    diag_matrix = torch.diag_embed(sig_t).unsqueeze(0).repeat(batch_size, 1, 1)
    return diag_matrix

def f(t, x, y, z):
    #CfD
    #c_0 = torch.ones(dim_x, device = device )*np.exp(x0_value)
    #value = (strike_t +  - x) * np.exp(-rho * t)

    #Benchmark example
    c_0 = torch.zeros(dim_x, device = device)
    value = 10*(c_0 - x)* np.exp(-rho * t)
    output: [batch_size, dim_y]
    return torch.mean(value, dim=-1, keepdim=True)
    #return value1

def g(x):
    return 0*torch.ones(x.shape[0], dim_y, device = device)

def lower_barrier(t,x):  #lower barrier = when player 2 stops
    return torch.ones(batch_size, dim_y, device=device)*(-l)*np.exp(- rho*t)
    #return torch.ones(batch_size, dim_y) * (-2) * np.exp(- rho * t)


def upper_barrier(t,x): #upper barrier = when player 1 stops
    return torch.ones(batch_size, dim_y, device=device)*(u)*np.exp(-rho*t)
    #return torch.ones(batch_size, dim_y) * (2) * np.exp(-rho * t)


def run_solver(equation, label, save_path):
    print(f"=== Running solver for: {label} ===")
    bsde_itr = BSDEiter(equation, dim_h)

    Y0_list = []
    y_list = []

    for i in range(1):
        print(f"{label}: iteration {i}")
        start_time = time.time()
        loss, y = bsde_itr.train_whole(batch_size, N, save_path, itr, multiplier)
        end_time = time.time()
        Y0_list.append(float(y[0, 0]))
        y_list.append(y.detach().cpu().numpy())
        print(f"{label}: iteration {i} took {(end_time - start_time)/60:.4f} minutes")

    # Save results
    with open(os.path.join(save_path, f"Y0_{label}.json"), 'w') as f:
        json.dump(Y0_list, f, indent=2)

    with open(os.path.join(save_path, f"loss_{label}.json"), 'w') as f:
        json.dump(loss, f, indent=2)

    return Y0_list, y_list

if mode == "Training":

    dim_x, dim_y, dim_d, dim_h, N, itr, batch_size = 1, 1, 1, 100, 50, 100, 2 ** 10
    multiplier = 5

    ###################################
    kappa = 1.5 + np.random.random(dim_x)
    kappa = kappa.tolist()

    beta = 1.0 + np.random.random(dim_x)
    beta = beta.tolist()

    mu = np.zeros(dim_x)
    mu = mu.tolist()

    sig = np.ones(dim_x)
    sig = sig.tolist()

    T = 1
    dt = T/N

    rho = 0
    u = 0.7 #upper barrier
    l = 0.7 #lower barrier
    T = 1
    x0_value= 0
    x_0 = torch.ones(dim_x)*x0_value
    strike = 0

################################################
    run_parameters = {
        "dim_x": dim_x,
        "dim_y": dim_y,
        "dim_d": dim_d,
        "dim_h": dim_h,
        "N": N,
        "itr": itr,
        "batch_size": batch_size,
        "multiplier": multiplier,
        "x0_value": x0_value,
        "kappa": kappa,
        "beta": beta,
        "mu": mu,
        "sig": sig,
        "rho": rho,
        "K": strike,
        "T": T,
        "u": u,
        "l": l,
    }

    with open(os.path.join(new_folder, "params.json"), "w") as h:
        json.dump(run_parameters, h, indent=2)

    x_0 = torch.tensor(x0_value, dtype=torch.float32, device=device)

    equation_explicit = fbsde(x_0, b, sigma, f, g, lower_barrier, upper_barrier,
                              T, dim_x, dim_y, dim_d)

    equation_empirical = fbsde(x_0, b_empirical, sigma, f, g, lower_barrier, upper_barrier,
                               T, dim_x, dim_y, dim_d)

    Y0_explicit, y_explicit = run_solver(equation_explicit, "Explicit", folder_explicit)
    Y0_empirical, y_empirical = run_solver(equation_empirical, "Empirical", folder_empirical)

else:
    import matplotlib.pyplot as plt
    import pandas as pd

    with open(os.path.join(new_folder, "params.json"), "r") as h:
        loaded_params = json.load(h)

    dim_x = loaded_params["dim_x"]
    dim_y = loaded_params["dim_y"]
    dim_d = loaded_params["dim_d"]
    dim_h = loaded_params["dim_h"]
    N = loaded_params["N"]
    itr = loaded_params["itr"]
    batch_size = loaded_params["batch_size"]
    multiplier = loaded_params["multiplier"]

    x0_value = loaded_params["x0_value"]

    kappa = loaded_params["kappa"]
    beta = loaded_params["beta"]
    mu = loaded_params["mu"]
    sig = loaded_params["sig"]

    rho = loaded_params["rho"]
    T = loaded_params["T"]
    u = loaded_params["u"]
    l = loaded_params["l"]


    #x_0 = torch.ones(dim_x, device=device)*x0_value
    x_0 = torch.tensor(x0_value, device=device)

    equation_explicit = fbsde(x_0, b, sigma, f, g, lower_barrier, upper_barrier,
                              T, dim_x, dim_y, dim_d)
    equation_empirical = fbsde(x_0, b_empirical, sigma, f, g, lower_barrier, upper_barrier,
                               T, dim_x, dim_y, dim_d)

    with open(folder_explicit + "loss_Explicit.json", 'r') as f:
        loss_exp = json.load(f)
    with open(folder_explicit + "Y0_Explicit.json", 'r') as f:
        Y0_exp = json.load(f)

    with open(folder_empirical + "loss_Empirical.json", 'r') as f:
        loss_emp = json.load(f)
    with open(folder_empirical + "Y0_Empirical.json", 'r') as f:
        Y0_emp = json.load(f)

    model_exp = Model(equation_explicit, dim_h)
    model_exp.eval()
    result = Result(model_exp, equation_explicit)

    model_emp = Model(equation_empirical, dim_h)
    model_emp.eval()
    result = Result(model_emp, equation_empirical)

    flag = True
    while flag:
        W = result.gen_b_motion(batch_size, N)
        x = result.gen_x(batch_size, N, W)
        flag = torch.isnan(x).any()

    ###########################
    # Brownian motion
    Wt = torch.cumsum(W, dim=-1)
    Wt = torch.roll(Wt, 1, -1)
    Wt[:, :, 0] = torch.zeros(batch_size, dim_d)
    ##########################################

    y, z = result.predict(N, batch_size, x, folder_explicit)

    y_emp, z_emp = result.predict(N, batch_size, x, folder_empirical)

    ############################################
    # Hitting times analysis

    # time
    t = torch.linspace(0, T, N)
    ############################

    x_np = x.detach().numpy()  # Shape: (batch_size, dim_x, N)
    y_np = y.detach().numpy()  # Shape: (batch_size, dim_y, N)
    z_np = z.detach().numpy()

    lower_np = lower_barrier(t, x).detach().numpy()
    upper_np = upper_barrier(t, x).detach().numpy()

    ####### PLOTS ###############################
    # loss analysis
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    # X-axes
    itr_ax_fine = np.linspace(1, itr * multiplier, itr * multiplier)
    itr_ax_coarse = np.linspace(1, itr, itr)

    # Subplot 1: Loss N-1
    axes[0, 0].plot(itr_ax_fine, loss_exp[0], label="Explicit")
    axes[0, 0].plot(itr_ax_fine, loss_emp[0], color='red', linestyle='--', label="Empirical")
    axes[0, 0].set_title("Loss at time step N-1")

    # Subplot 2: Loss N-2
    axes[0, 1].plot(itr_ax_fine, loss_exp[1], label="Explicit")
    axes[0, 1].plot(itr_ax_fine, loss_emp[1], color='red', linestyle='--', label="Empirical")
    axes[0, 1].set_title("Loss at time step N-2")

    # Subplot 3: Loss N-3
    axes[1, 0].plot(itr_ax_coarse, loss_exp[2], label="Explicit")
    axes[1, 0].plot(itr_ax_coarse, loss_emp[2], color='red', linestyle='--', label="Empirical")
    axes[1, 0].set_title("Loss at time step N-3")

    # Subplot 4: Loss N-5
    axes[1, 1].plot(itr_ax_coarse, loss_exp[5], label="Explicit")
    axes[1, 1].plot(itr_ax_coarse, loss_emp[5], color='red', linestyle='--', label="Empirical")
    axes[1, 1].set_title("Loss at time step N-5")

    for ax in axes.flat:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(graph_path, "loss_grid_comparison.png"))
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(itr_ax_coarse, loss_exp[2], color ='blue', label='Explicit')
    plt.plot(itr_ax_coarse, loss_emp[2], color='red', linestyle='--', label='Empirical')
    plt.savefig(graph_path + "loss_N-3.png")
    plt.show()
    plt.close()

####### Y0

    plt.figure(figsize=(10, 6))
    # Plot Explicit
    plt.hist(Y0_exp, bins=10, alpha=0.5, label='Explicit', color='blue')
    # Plot Empirical on top
    plt.hist(Y0_emp, bins=10, alpha=0.5, label='Empirical', color='red')
    # Means
    mean_Y0_exp = np.mean(Y0_exp)
    mean_Y0_emp = np.mean(Y0_emp)
    plt.axvline(mean_Y0_exp, color='blue', linestyle='dashed', linewidth=2,
                label=f'Mean Explicit = {mean_Y0_exp:.2f}')
    plt.axvline(mean_Y0_emp, color='red', linestyle='dashed', linewidth=2,
                label=f'Mean Empirical = {mean_Y0_emp:.2f}')

    plt.grid(True)
    plt.legend()
    plt.xlabel("Y0 values")
    plt.ylabel("Frequency")
    #plt.title("Histogram of Y0: Explicit vs Empirical")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_path, "Y0_hist_comparison.png"))
    plt.show()
    plt.close()

    j = np.random.randint(batch_size)

    plt.figure(figsize=(12, 6))
    for i in range(dim_x):
        plt.plot(t, x_np[j, i, :], label=f"x[{i}]")

    plt.title("All Dimensions of x over Time (Sample {})".format(j))
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path + f"x_{j}.png")
    plt.show()
    plt.close()

    j = np.random.randint(batch_size)
    k = np.random.randint(batch_size)

    random_indices = np.random.choice(batch_size, size=2, replace=False)
    colors = ['red', 'blue', 'green', 'yellow']  # add more colors if you have more y realizations
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i, idx in enumerate(random_indices):
        ax.plot(t, y[idx, 0, :].detach().numpy(), color=colors[i], linestyle='-', label=f"Y realization {i}")
        ax.plot(t, y_emp[idx, 0, :].detach().numpy(), color=colors[i], linestyle='--', label=f"Y realization {i}")

    # Plot upper and lower barriers (use j from np.random if you want consistency)
    ax.plot(t, upper_np[j, :], color="black", linestyle='--', label="Upper barrier")
    ax.plot(t, lower_np[j, :], color="green", linestyle='--', label="Lower barrier")
    # Add legend and title
    #ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(graph_path + str(j))
    plt.show()
