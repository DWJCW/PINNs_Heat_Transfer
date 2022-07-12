import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from cond import *

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
    """
    exact = pd.read_csv("exact2.csv")
    Umag = ((exact["U:0"]**2+exact["U:1"]**2+exact["U:2"]**2)**0.5).to_numpy()
    T = exact["T"].to_numpy()
    # p = exact["p"].to_numpy()
    x = exact["Points:0"].to_numpy()
    x_pred = np.linspace(-1, 1, len(x))
    y_pred = np.zeros_like(x_pred)
    t_pred = np.ones_like(x_pred)*5.0
    X_pred = np.vstack([x_pred, y_pred, t_pred]).T
    """
    dt = 1e-2
    ts = np.arange(0, 1+dt, dt)
    # plot PINNs

    # Load model
    N_b = 1000
    N_F = 1000
    N_x = 500
    N_y = 500
    N_t = 1000
    N_r = 100
    data = Data(N_b, N_F, N_x, N_y, N_t)
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = PINNs(data, layers)
    model.dnn = torch.load(SAVE_PATH)

    nts = [0, 2, 4, 8, 100]
    Ts = np.load("Ts.npy")

    for i in range(5):
        fig = plt.figure(figsize=[6, 2])
        ax = fig.add_subplot(111)
        nt = nts[i]
        time = ts[nt]
        #print(time)

        x_pred = np.linspace(0., 1., 100)
        y_pred = np.linspace(0.5, 0.5, 100)
        t_pred = np.linspace(1., 1., 100)*time
        X_pred = np.vstack([y_pred, x_pred, t_pred]).T

        T_pred = model.predict(X_pred).detach().cpu().numpy()
        #print(X_pred)
        ax.plot(x_pred, T_pred, "r--", label="Prediction")

        # plot exact
        T_exact = Ts[nt]
        Nx = len(T_exact)
        x_plot = np.linspace(0, 1, Nx)
        ax.plot(x_plot, T_exact[-1::-1, int(Nx/2)], 'b-', label="Exact")
        ax.set_title("time = {} s".format(time))
        ax.set_ylim([0, 0.4])
        ax.set_xlabel(r"$x/\mathrm{m}$")
        ax.set_ylabel("T")
        ax.legend()
        # ax.set_aspect(1)
        fig.tight_layout()
        fig.savefig("img/line{}.png".format(i))
        
    # plt.show()