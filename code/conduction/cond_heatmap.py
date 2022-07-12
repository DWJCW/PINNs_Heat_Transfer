from cond import *
import torch
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
from pyDOE import lhs
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from matplotlib import cm

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

if __name__=="__main__":
    print(SAVE_PATH)
    N_b = 1
    N_F = 1
    N_x = 1
    N_y = 1
    N_t = 1
    N_r = 1
    data = Data(N_b, N_F, N_x, N_y, N_t)
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = PINNs(data, layers)
    model.dnn = torch.load(SAVE_PATH)

    # Calc
    N_x = 50
    N_y = 50
    N_t = 1

    x_lim = data.x_lim
    y_lim = data.y_lim
    t_lim = data.t_lim

    lb = np.array([x_lim[0], y_lim[0], t_lim[0]])
    ub = np.array([x_lim[1], y_lim[1], t_lim[1]])


    dt = 1e-2
    ts = np.arange(0, 1+dt, dt)

    nt = [0, 5, 10, 50, 100]

    Ts = np.load("Ts.npy")
    for i in range(5):
        fig = plt.figure(figsize=(6, 2.0))

        time = ts[nt[i]]
        x = np.linspace(x_lim[0], x_lim[1], N_x)
        y = np.linspace(y_lim[0], y_lim[1], N_y)
        # t = np.linspace(t_lim[0], t_lim[1], N_t)
        t = np.linspace(time, time, N_t)

        X, Y, T = np.meshgrid(x, y, t)
        X_pred = np.hstack( 
            (X.flatten()[:,None], Y.flatten()[:,None], T.flatten()[:,None])
        )
        T_pred = model.predict(X_pred).detach().cpu().numpy().reshape(N_x, N_y)

        ax = fig.add_subplot(1, 2, 1)
        ax.plot([0.5, 0.5], [0.0, 1.0], color='white')
        h = ax.imshow(T_pred, interpolation='nearest', cmap=cm.jet, 
                    extent=[x.min(), x.max(), y.min(), y.max()], vmin=0, vmax=0.35,
                    origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        # cbar.ax.tick_params(labelsize=15)

        ax.set_xlabel(r'$x/\mathrm{m}$')
        ax.set_ylabel(r'$y/\mathrm{m}$')
        ax.set_title('${}(x,y,t)$ prediction'.format("T"))
        ax.set_aspect(1)

        T_exact = Ts[nt[i]]
        ax = fig.add_subplot(1, 2, 2)
        h = ax.imshow(T_exact[-1::-1, :], interpolation='nearest', cmap=cm.jet, 
                    extent=[x.min(), x.max(), y.min(), y.max()], vmin=0, vmax=0.35,
                    origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        # cbar.ax.tick_params(labelsize=15)
        ax.set_xlabel(r'$x/\mathrm{m}$')
        ax.set_ylabel(r'$y/\mathrm{m}$')
        ax.set_title('${}(x,y,t)$ exact'.format("T"))
        ax.set_aspect(1)

        # fig.suptitle("time = {:.2f} s".format(time))

        fig.tight_layout()
        fig.savefig("img/{}.png".format(i))
    # plt.show()