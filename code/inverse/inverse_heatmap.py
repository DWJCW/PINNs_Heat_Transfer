from inverse import Data, PINNs, DNN, SAVE_PATH
import torch
from matplotlib import pyplot as plt
import numpy as np
# from collections import OrderedDict
# from pyDOE import lhs
# from scipy.interpolate import griddata
# import matplotlib.gridspec as gridspec
from matplotlib import cm
# from mpl_toolkits.axes_grid1 import make_axes_locatable

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['savefig.dpi'] = 300

if __name__ == "__main__":
    print(SAVE_PATH)
    N_F = 1
    N_known = 1
    test_alpha = 0.2
    data = Data(N_F, N_known, test_alpha)
    layers = [2, 10, 20, 40, 40, 40, 20, 10, 5, 1]
    model = PINNs(data, layers)
    model.dnn = torch.load(SAVE_PATH)

    # Plot
    N_x = 100
    N_t = 100
    x_lim = data.x_lim
    t_lim = data.t_lim
    fig = plt.figure(figsize=(8, 4))
    x = np.linspace(x_lim[0], x_lim[1], N_x)
    t = np.linspace(t_lim[0], t_lim[1], N_t)
    X, T = np.meshgrid(x, t)
    X_pred = np.hstack(
        (X.flatten()[:, None], T.flatten()[:, None])
    )
    T_pred = model.predict(X_pred).detach().cpu().numpy().reshape(N_t, N_x)
    ax = fig.add_subplot(1, 1, 1)
    h = ax.imshow(T_pred.T, interpolation='nearest', cmap=cm.jet,
                  extent=[t.min(), t.max(), x.min(), x.max()], vmin=0,
                  vmax=1., origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.10)
    # cbar = fig.colorbar(h, cax=cax)
    cbar = fig.colorbar(h)
    ax.set_aspect(1/4)
    plt.show()
