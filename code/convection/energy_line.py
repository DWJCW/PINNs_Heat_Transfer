import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from energy import *

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['savefig.dpi'] = 300

if __name__ == "__main__":
    I = 2
    exact = pd.read_csv("exact{}.csv".format(I))
    Umag = ((exact["U:0"]**2+exact["U:1"]**2+exact["U:2"]**2)**0.5).to_numpy()
    T = exact["T"].to_numpy()
    # p = exact["p"].to_numpy()
    x = exact["Points:0"].to_numpy()
    x_pred = np.linspace(-1, 1, len(x))
    y_pred = np.zeros_like(x_pred)
    t_pred = np.ones_like(x_pred)*5.0
    X_pred = np.vstack([x_pred, y_pred, t_pred]).T

    # Load model
    N_b = 1000
    N_F = 1000
    N_x = 500
    N_y = 500
    N_t = 1000
    N_r = 100
    data = Data(N_b, N_F, N_x, N_y, N_t)
    layers_U = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    layers_T = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = PINNs(data, layers_U, layers_T)
    model.dnn_T = torch.load("conv.pt")

    u_pred, v_pred, p_pred, T_pred = model.predict(X_pred)
    u_pred = u_pred.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    # p_pred = p_pred.detach().cpu().numpy()
    T_pred = T_pred.detach().cpu().numpy()
    Umag_pred = np.sqrt(np.square(u_pred)+np.square(v_pred))
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(121)
    ax.plot(x-1, T, 'b-', label="Exact")
    ax.plot(x_pred, T_pred, "r--", label="Prediction")
    # ax.set_ylim([1, 2])
    ax.set_xlim([-1.0, 1.0])
    #ax.set_ylim([0.4, 1.1])
    ax.set_title("Temperature")
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.2, -0.2), 
        ncol=5, 
        frameon=False, 
    )
    ax.set_xlabel(r'$x/\mathrm{m}$')
    ax.set_ylabel(r'$T$')
    fig.tight_layout()

    ax = fig.add_subplot(122)
    ax.plot(x-1, Umag, 'b-', label="Exact")
    ax.plot(x_pred, Umag_pred, "r--", label="Prediction")
    # ax.set_ylim([1, 2])
    ax.set_xlim([-1.0, 1.0])
    #ax.set_ylim([0.9, 1.6])
    ax.set_title("Velocity Magnitude")
    ax.set_xlabel(r'$x/\mathrm{m}$')
    ax.set_ylabel(r'$U_\mathrm{mag}$')
    #ax.legend()

    fig.tight_layout()
    fig.savefig("img/line{}.png".format(I))
    plt.show()