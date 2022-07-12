from ns import *
import torch
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
from pyDOE import lhs
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib import cm

if __name__=="__main__":
    N_b = 1000
    N_F = 1000
    N_x = 500
    N_y = 500
    N_t = 1000
    N_r = 100
    data = Data(N_b, N_F, N_x, N_y, N_t)
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    model = PINNs(data, layers)
    model.dnn = torch.load("ns1.pt")

    # Calc
    N_x = 100
    N_y = 100
    N_t = 1

    x_lim = data.x_lim
    y_lim = data.y_lim
    t_lim = data.t_lim

    lb = np.array([x_lim[0], y_lim[0], t_lim[0]])
    ub = np.array([x_lim[1], y_lim[1], t_lim[1]])

    time = 5.0

    x = np.linspace(x_lim[0], x_lim[1], N_x)
    y = np.linspace(y_lim[0], y_lim[1], N_y)
    # t = np.linspace(t_lim[0], t_lim[1], N_t)
    t = np.linspace(time, time, N_t)

    X, Y, T = np.meshgrid(x, y, t)
    X_pred = np.hstack( 
        (X.flatten()[:,None], Y.flatten()[:,None], T.flatten()[:,None])
    )
    u_pred, v_pred, p_pred = model.predict(X_pred)
    u_pred = u_pred.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    p_pred = p_pred.detach().cpu().numpy()
    U_pred = u_pred.reshape(N_x, N_y)
    V_pred = v_pred.reshape(N_x, N_y)
    Umag_pred = np.sqrt(U_pred**2+V_pred**2)
    P_pred = p_pred.reshape(N_x, N_y)

    # plot

    """ The aesthetic setting has changed. """

    ####### Row 0: u(t,x) ##################    

    fig = plt.figure(figsize=(12, 16))
    # fig.suptitle("$\\alpha={:.2f}$".format(alpha_value), fontsize = 24)

    ###### Prediction U#####################
    plot_data = [U_pred, V_pred, Umag_pred, P_pred]
    plot_name = ["u", "v", "Umag", "p"]

    # plot_data = [U_pred, V_pred]
    # plot_name = ["u", "v"]

    for i in range(len(plot_data)):
        ax = fig.add_subplot(411+i)
        h = ax.imshow(plot_data[i], interpolation='nearest', cmap=cm.jet, 
                    extent=[x.min(), x.max(), y.min(), y.max()], 
                    origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        # font size doubled
        ax.set_title('${}(x,y,t),t={:.2f}$ prediction'.format(plot_name[i], time))
        ax.set_aspect(1)
    fig.tight_layout()
    plt.show()