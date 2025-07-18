import torch
import numpy as np
import os
import json

from networkx.utils.decorators import np_random_state

from DRBSDE import fbsde
from DRBSDE import BSDEiter
from DRBSDE import Model
from DRBSDE import Result




path = "state_dicts/"


new_folder_flag = True
new_folder = "Test2d_2/"

if new_folder_flag:
    path = new_folder + path
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(new_folder + "Graphs"):
        os.makedirs(new_folder + "Graphs")
        #path = new_folder + path
    graph_path = new_folder + "Graphs/"
ref_flag = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


mode = "Training"
mode = "Testing"
ht_analysis = False


def b(t, x):
    # x: shape [batch_size, dim_x]
    mu_t = torch.tensor(mu, device = device)
    kappa_t = torch.tensor(kappa, device = device)

    drift = kappa_t*(mu_t - x)
    # X: OU process
    return drift

def sigma(t, x):
    # x: shape [batch_size, dim_x, dim_d]
    sig_t = torch.tensor(sig, device = device)
    diag_matrix = torch.diag_embed(sig_t).unsqueeze(0).repeat(batch_size, 1, 1)

    return diag_matrix


def f(t, x, y, z):
    c_0 = torch.ones(dim_x, device = device )*np.exp(x0_value)
    value = (c_0 - x)* np.exp(-rho * t)
    # output: [batch_size, dim_y]
    return torch.mean(value, dim=-1, keepdim=True)
    #return value1


def g(x):
    return 0*torch.ones(x.shape[0], dim_y, device = device)


def lower_barrier(t,x):  #lower barrier = when player 2 stops
    return torch.ones(batch_size, dim_y, device=device)*(-l)*np.exp(- rho*t)
    #return torch.ones(batch_size, dim_y) * (-2) * np.exp(- rho * t)


def upper_barrier(t,x): #upper barrier = when player 1 stops
    return torch.ones(batch_size, dim_y, device=device)*(l)*np.exp(-rho*t)
    #return torch.ones(batch_size, dim_y) * (2) * np.exp(-rho * t)



if mode == "Training":

    dim_x, dim_y, dim_d, dim_h, N, itr, batch_size = 20, 1, 20, 50, 50, 50, 2 ** 10
    multiplier = 5

    ###################################
    # CfD EXAMPLE
    r = 0.04
    R = 0.06

    x0_value = 4.35  # initial price
    x0_value1 = 4.35  # initial price

    #kappa = np.random.random(dim_x)*0.1 + 23.667704403397515
    kappa = np.ones(dim_x)*23.667704403397515
    kappa = kappa.tolist()

    #mu = np.random.random(dim_x)*0.1 +4.331928194132446
    mu = np.ones(dim_x)*4.331928194132446
    mu = mu.tolist()

    #sig = np.random.random(dim_x)*0.4321650311213917
    sig = np.ones(dim_x) * 0.4321650311213917
    sig = sig.tolist()

    kappa1 = kappa
    mu1 = mu
    sig1 = sig



    c_0 = np.exp(x0_value)
    # c_0 = x0_value

    rho = 0.04
    T = 1
    u = 1.56  # upper barrier
    l = 0.31  # lower barrier

    ############################
    #BENCHMARK EXAMPLE
    '''
    kappa = 2.0  # Speed of mean reversion
    mu = 0.0     # Long-term mean price (USD/MWh)
    sig = 1.0  # Volatility (USD/MWh per sqrt(day))
    K = 0.0      # Strike price (USD/MWh)
    Q = 10      # Quantity (MWh)
    T = 1        # Simulation length (days)
    dt = T/N
    #the payoff function will be Q*(K-P)=c0 + c1*P
    c_0 = K*Q
    c_1 = -Q
    rho = 0
    gamma1 = 2 #upper barrier
    gamma2 = 2 #lower barrier
    T = 1
    x0_value= 0
    x_0 = torch.ones(dim_x)*x0_value
    '''

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
        "x0_value1": x0_value1,
        "kappa": kappa,
        "mu": mu,
        "sig": sig,
        "kappa1": kappa1,
        "mu1": mu1,
        "sig1": sig1,
        "c_0": c_0,
        "rho": rho,
        "T": T,
        "u": u,
        "l": l,
    }

    x_0 = torch.ones(dim_x, device=device)*x0_value
    #x_0[0] = x0_value
    #x_0[1] = x0_value1

    #Define the FBSDE system
    equation = fbsde(x_0, b, sigma, f, g, lower_barrier, upper_barrier, T, dim_x, dim_y, dim_d)

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "params.json"), "w") as h:
        json.dump(run_parameters, h, indent=2)


    bsde_itr = BSDEiter(equation, dim_h)
    Y0=[]
    f1 = []
    f2 = []
    for i in range(1):
        print(f"iteration n {i}")
        loss, y  = bsde_itr.train_whole(batch_size, N, path, itr, multiplier)
        Y0.append(float(y[0, 0]))

    with open(path + "loss.json", 'w') as f:
        json.dump(loss, f, indent=2)


else:
    import matplotlib.pyplot as plt

    with open(os.path.join(path, "params.json"), "r") as h:
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
    x0_value1 = loaded_params["x0_value1"]
    kappa = loaded_params["kappa"]
    mu = loaded_params["mu"]
    sig = loaded_params["sig"]
    kappa1 = loaded_params["kappa1"]
    mu1 = loaded_params["mu1"]
    sig1 = loaded_params["sig1"]
    c_0 = loaded_params["c_0"]
    rho = loaded_params["rho"]
    T = loaded_params["T"]
    u = loaded_params["u"]
    l = loaded_params["l"]



    x_0 = torch.ones(dim_x, device=device)*x0_value

    equation = fbsde(x_0, b, sigma, f, g, upper_barrier, lower_barrier, T, dim_x, dim_y, dim_d)

    with open(path + "loss.json", 'r') as f:
        loss = json.load(f)

    model = Model(equation, dim_h)
    model.eval()
    result = Result(model, equation)

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

    y, z = result.predict(N, batch_size, x, path)

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
    axes[0, 0].plot(itr_ax_fine, loss[0])
    axes[0, 0].set_title("Loss at time step N-1")
    # Subplot 2: Loss N-2
    axes[0, 1].plot(itr_ax_fine, loss[1])
    axes[0, 1].set_title("Loss at time step N-2")
    # Subplot 3: Loss N-3
    axes[1, 0].plot(itr_ax_coarse, loss[2])
    axes[1, 0].set_title("Loss at time step N-3")
    # Subplot 4: Loss N-5
    axes[1, 1].plot(itr_ax_coarse, loss[5])
    axes[1, 1].set_title("Loss at time step N-5")
    for ax in axes.flat:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(graph_path + "loss_grid.png")
    plt.show()

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

    j = np.random.randint(batch_size)
    fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
    # Plot Y_1 (dim 0)
    plt.plot(t, y[j, 0, :].detach().numpy(), color="red", label="Y_1")
    plt.plot(t, upper_np[j,:], color="blue")
    plt.plot(t, lower_np[j,:], color="green")
    #axes[0].legend(loc="upper right")
    plt.title("Y")
    plt.tight_layout()
    plt.savefig(graph_path + str(j))
    plt.show()

if ht_analysis:
    hitting_times_lower1 = []
    actual_hitting_times_lower1 = []
    prices_at_lower_stopping1 = []

    hitting_times_upper1 = []
    actual_hitting_times_upper1 = []
    prices_at_upper_stopping1 = []

    hitting_times_lower2 = []
    actual_hitting_times_lower2 = []
    prices_at_lower_stopping2 = []

    hitting_times_upper2 = []
    actual_hitting_times_upper2 = []
    prices_at_upper_stopping2 = []

    for j in range(batch_size):

        y1_numpy = y[j,0 :].detach().numpy()

        x_numpy = x[j, 0, :].detach().numpy()  # Price process for batch element i

        # Check lower stopping barrier for y1
        hit_lower1 = (y1_numpy <= lower_np[j, :]).nonzero()[0]
        if len(hit_lower1) > 0:
            hit_time1 = hit_lower1[0].item()
            hitting_times_lower1.append(hit_time1)
            actual_hitting_times_lower1.append(hit_time1)
            prices_at_lower_stopping1.append(x_numpy[hit_time1])  # Store price at hitting time
        else:
            hitting_times_lower1.append(N)

        # Check upper stopping barrier for y1
        hit_upper1 = (y1_numpy >= upper_np[j, :]).nonzero()[0]
        if len(hit_upper1) > 0:
            hit_time1 = hit_upper1[0].item()
            hitting_times_upper1.append(hit_time1)
            actual_hitting_times_upper1.append(hit_time1)
            prices_at_upper_stopping1.append(x_numpy[hit_time1])  # Store price at hitting time
        else:
            hitting_times_upper1.append(N)

    hitting_times_lower1 = np.array(hitting_times_lower1)
    hitting_times_upper1 = np.array(hitting_times_upper1)

    actual_hitting_times_lower1 = np.array(actual_hitting_times_lower1)
    actual_hitting_times_upper1 = np.array(actual_hitting_times_upper1)

    hitting_times_lower_time1 = hitting_times_lower1 / N
    hitting_times_upper_time1 = hitting_times_upper1 / N

    actual_hitting_times_upper_time1 = actual_hitting_times_upper1 / N
    actual_hitting_times_lower_time1 = actual_hitting_times_lower1 / N

    number_exit_times_lower1 = len(actual_hitting_times_lower1)
    number_exit_times_upper1 = len(actual_hitting_times_upper1)

    plt.hist(actual_hitting_times_lower_time1, bins=10, alpha=0.5, label=r'Exit times for Player 1')
    plt.hist(actual_hitting_times_upper_time1, bins=10, alpha=0.5, label=r'Exit times for Player 0')
    plt.legend()
    plt.grid(True)
    plt.savefig(graph_path + "hitting_times")
    plt.show()


'''
for j in range(batch_size):

        y_numpy = y[j, 0, :].detach().numpy()
        x_numpy = x[j, 0, :].detach().numpy()  # Price process for batch element i

        # Check lower stopping barrier
        hit_lower = (y_numpy <= l2_vals[:,j,0]).nonzero()[0]
        if len(hit_lower) > 0:
            hit_time = hit_lower[0].item()
            hitting_times_lower.append(hit_time)
            actual_hitting_times_lower.append(hit_time)
            prices_at_lower_stopping.append(x_numpy[hit_time])  # Store price at hitting time
        else:
            hitting_times_lower.append(N)

        # Check upper stopping barrier
        hit_upper = (y_numpy >= l1_vals[:,j,0]).nonzero()[0]
        if len(hit_upper) > 0:
            hit_time = hit_upper[0].item()
            hitting_times_upper.append(hit_time)
            actual_hitting_times_upper.append(hit_time)
            prices_at_upper_stopping.append(x_numpy[hit_time])  # Store price at hitting time
        else:
            hitting_times_upper.append(N)

    hitting_times_lower = np.array(hitting_times_lower)
    hitting_times_upper = np.array(hitting_times_upper)

    actual_hitting_times_lower = np.array(actual_hitting_times_lower)
    actual_hitting_times_upper = np.array(actual_hitting_times_upper)

    hitting_times_lower_time = hitting_times_lower / N
    hitting_times_upper_time = hitting_times_upper / N

    actual_hitting_times_upper_time = actual_hitting_times_upper / N
    actual_hitting_times_lower_time = actual_hitting_times_lower / N

    # Percent of times player 1/2 stops
    f1.append(len(actual_hitting_times_upper) / batch_size)
    f2.append(len(actual_hitting_times_lower) / batch_size)




Y0_np = np.array(Y0)
np.save(new_folder + "Y0.npy", Y0_np)
#Y0_np = np.load(new_folder + "Y0.npy")


plt.hist(Y0_np)
plt.vlines(np.mean(Y0_np),0,20,colors="red")
plt.grid(True)
plt.savefig(path + "Y0_hist")
plt.show()

std_Y0 = np.std(Y0)
print(f'Standard Deviation of Y0: {std_Y0:.4f}')

f1_np = np.array(f1)
f2_np = np.array(f2)
np.save(new_folder + "f1.npy", f1_np)
np.save(new_folder + "f2.npy", f2_np)

plt.hist(f1_np)
plt.grid(True)
plt.vlines(np.mean(f1_np),0,20,colors="red")
plt.savefig(path + "f1_hist")
plt.show()


plt.hist(f2_np)
plt.grid(True)
plt.vlines(np.mean(f2_np),0,20,colors="red")
plt.savefig(path + "f2_hist")
plt.show()




##################################################################
#Explicit solutions


time = torch.unsqueeze(t, dim=0)
time = torch.unsqueeze(time, dim=0)
time = torch.repeat_interleave(time, repeats=batch_size, dim=0)

term1 = (c_0 + c_1 * mu) * (T - time)
term2 = c_1 / kappa * (x_0 - mu) * (np.exp(-kappa * T) - torch.exp(-kappa * time))

exp_kappa_t = torch.exp(kappa * t)  # Shape: (N,)
exp_neg_kappa_t = torch.exp(-kappa * t)  # Shape: (N,)

inner_integral = torch.zeros(batch_size, dim_d, N)  # To store results for all t
outer_integral = torch.zeros(batch_size, dim_d, N)

for i in range(N):  # For each time t[i]
    inner_integral[:, :, i] = torch.sum(exp_kappa_t[:i] * W[:, :, :i], dim=-1)
    outer_integral[:, :, i] = torch.sum(exp_neg_kappa_t[i-1:] * (T / N))

ytrue = term1 + term2 + sig * c_1 * outer_integral* inner_integral

#
#
# term1 = (c_0 + c_1 * mu) * (T - time)
# term2 = - c_1 / kappa * (x_0 - mu) * ( np.exp(-kappa * T) - torch.exp(-kappa * time))
#
# exp_kappa_t = torch.exp(kappa * t)  # Shape: (N,)
# exp_neg_kappa_t = torch.exp(-kappa * t)  # Shape: (N,)
#
# inner_integral = torch.zeros(batch_size, dim_d, N)  # To store results for all t
# outer_integral =  sig * c_1 /kappa*(  torch.exp(-kappa * time) - np.exp(-kappa * T))
#
# for i in range(N):  # For each time t[i]
#     inner_integral[:, :, i] = torch.sum(exp_kappa_t[:i] * W[:, :, :i], dim=-1)
#
# ytrue = term1 + term2 + outer_integral*inner_integral

j = np.random.randint(batch_size)

plt.plot(t,x[j,0,:].detach().numpy(), color="red", label="P_t")
plt.title("Electricity market price P")
plt.legend()
plt.show()



j = np.random.randint(batch_size)
plt.plot(t, y[j, 0, :].detach().numpy(), color="red", label="DRBSDE")
#plt.plot(t,ytrue[j,0,:], color="blue", label="Analytical")

plt.plot(t, l2_vals[:, 0], color="green", linestyle="dotted", label= r'f_2(t,X_t)')
plt.plot(t, l1_vals[:, 0], color="purple", linestyle="dashed", label= r'f_1(t,X_t)' )
#plt.plot(t, ytrue[j, 0, :], color="blue", label="Analytical")

plt.legend(loc=(0.75,0.7))
plt.savefig(graph_path + str(j))
plt.show()



j_indices = np.random.randint(batch_size, size=2)  # 3 random realizations
colors = ["red", "blue", "brown"]
for idx, j in enumerate(j_indices):
    label = f"Y_t realization {j_indices[idx]}" #if idx == 0 else None  # only label the first curve
    plt.plot(t, y[j, 0, :].detach().numpy(), color=colors[idx], label=label)
plt.plot(t, l2_vals[:, 0], color="green", linestyle="dotted", label=r'-f_2(t,X_t)')
plt.plot(t, l1_vals[:, 0], color="purple", linestyle="dashed", label=r'f_1(t,X_t)')

# Show legend (optional: comment out to hide legend)
plt.legend(loc=(0.75, 0.7))

plt.savefig(graph_path + str(j_indices[0]))
plt.show()


#################################
#loss analysis

#graph a loss at specific time
itr_ax = np.linspace(1,itr*multiplyer,itr*multiplyer)

plt.plot(itr_ax[:],loss[0][:])
#plt.title("Loss at time step N-1")
plt.savefig(graph_path + "loss N-1")
plt.show()

plt.plot(itr_ax[:],loss[1][:])
#plt.title("Loss at time step N-2")
plt.savefig(graph_path + "loss N-2")
plt.show()

itr_ax = np.linspace(1,itr,itr)


plt.plot(itr_ax[:],loss[2][:])
#plt.title("Loss at time step N-3")
plt.savefig(graph_path + "loss N-3")
plt.show()

plt.plot(itr_ax[:],loss[5][:])
plt.title("Loss at N-5")
plt.show()


plt.plot(itr_ax[:],loss[-2][:])
plt.title("Loss at N-5")
plt.show()



plt.hist(actual_hitting_times_lower_time, bins=10, alpha=0.5, label=r'Exit times for Player 2')
plt.hist(actual_hitting_times_upper_time, bins=10, alpha=0.5,  label=r'Exit times for Player 1')
plt.legend()
plt.grid(True)
plt.savefig(graph_path + "hitting_times")
plt.show()


#plt.figure(figsize=(8,5))
plt.scatter(actual_hitting_times_lower_time, prices_at_lower_stopping, color='blue', alpha=0.5, label="Player 2 Stops")
plt.scatter(actual_hitting_times_upper_time, prices_at_upper_stopping, color='red', alpha=0.5, label="Player 1 Stops")
plt.xlabel("Time")
plt.ylabel("Electricity Price")
plt.legend(loc="lower right")
plt.savefig(graph_path + "times_price_scatter")
plt.show()







import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

j = np.random.randint(batch_size)

stop_time_1 = hitting_times_upper[j]
stop_time_2 = hitting_times_lower[j]

stop_time_1 = int(np.round(stop_time_1))
stop_time_2 = int(np.round(stop_time_2))

axes[0].plot(t, x[j, 0, :].detach().numpy(), color="red", label="Electricity Price P")
axes[0].set_title("Electricity Market Price P with Exit Points")

if stop_time_1 < len(t):  # Ensures it is a valid index
    axes[0].scatter(t[stop_time_1], x[j, 0, stop_time_1].detach().numpy(), color="blue", label="Player 1 Exit", zorder=3)

if stop_time_2 < len(t):
    axes[0].scatter(t[stop_time_2], x[j, 0, stop_time_2].detach().numpy(), color="green", label="Player 2 Exit", zorder=3)

axes[0].set_xlabel("Time")
axes[0].set_ylabel("Electricity Price \( P_t \)")
axes[0].legend()
axes[0].grid()

axes[1].plot(t, y[j, 0, :].detach().numpy(), color="red", label="DRBSDE")
axes[1].plot(t, l2_vals[:, 0], color="green", linestyle="dashed", label=r'$f_2(t,X_t)$')
axes[1].plot(t, l1_vals[:, 0], color="purple", linestyle="dotted", label=r'$f_1(t,X_t)$')
    # axes[1].plot(t, ytrue[j, 0, :], color="blue", label="Analytical")

axes[1].set_title("DRBSDE and Boundaries")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Value Function / Boundaries")
axes[1].legend(loc="lower right")
axes[1].grid()

    # Display the combined plot
plt.tight_layout()
plt.show()

#TO SET BARRIERS
y_max = torch.max(y, dim=-1).values
y_90 = float(torch.quantile(y_max,0.9,dim=0))

y_min = torch.min(y, dim=-1).values
y_10 = float(torch.quantile(y_min,0.2,dim=0))


y_max= np.max(y.detach().numpy(),axis=-1)


print("Percentage of exits for player 1: ", len(actual_hitting_times_upper) / batch_size)
print("Percentage of exits for player 2: ", len(actual_hitting_times_lower) / batch_size)


'''









