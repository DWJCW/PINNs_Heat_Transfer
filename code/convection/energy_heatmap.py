from energy import *
import torch
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
from pyDOE import lhs
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

    # Calc
    N_x = 100
    N_y = 100
    N_t = 1

    x_lim = data.x_lim
    y_lim = data.y_lim
    t_lim = data.t_lim

    lb = np.array([x_lim[0], y_lim[0], t_lim[0]])
    ub = np.array([x_lim[1], y_lim[1], t_lim[1]])

    times = [0.2, 0.4, 0.6, 0.8, 1.0, 5.0]
    times = np.arange(0, 5.0, 0.1)
    vmins = [0., -0.25, 1.5, 2.0, 1.0]
    vmaxs = [1.5, 0.25, 0.0, 0.0, 0.0]
    for i in range(len(times)):
        time = times[i]
        x = np.linspace(x_lim[0], x_lim[1], N_x)
        y = np.linspace(y_lim[0], y_lim[1], N_y)
        # t = np.linspace(t_lim[0], t_lim[1], N_t)
        t = np.linspace(time, time, N_t)

        X, Y, T = np.meshgrid(x, y, t)
        X_pred = np.hstack( 
            (X.flatten()[:,None], Y.flatten()[:,None], T.flatten()[:,None])
        )
        u_pred, v_pred, p_pred, T_pred = model.predict(X_pred)
        u_pred = u_pred.detach().cpu().numpy()
        v_pred = v_pred.detach().cpu().numpy()
        p_pred = p_pred.detach().cpu().numpy()
        T_pred = T_pred.detach().cpu().numpy()
        U_pred = u_pred.reshape(N_x, N_y)
        V_pred = v_pred.reshape(N_x, N_y)
        Umag_pred = np.sqrt(U_pred**2+V_pred**2)
        P_pred = p_pred.reshape(N_x, N_y)
        T_pred = T_pred.reshape(N_x, N_y)
        

        # plot

        """ The aesthetic setting has changed. """

        ####### Row 0: u(t,x) ##################    

        fig = plt.figure(figsize=(6, 9))
        # fig.suptitle("$\\alpha={:.2f}$".format(alpha_value), fontsize = 24)

        ###### Prediction U#####################
        plot_data = [U_pred, V_pred, Umag_pred, P_pred, T_pred]
        plot_name = ["u", "v", "Umag", "p", "T"]

        # plot_data = [U_pred, V_pred]
        # plot_name = ["u", "v"]

        for j in range(len(plot_data)):
            ax = fig.add_subplot(511+j)
            h = ax.imshow(plot_data[j], interpolation='nearest', cmap=cm.jet, 
                        extent=[x.min(), x.max(), y.min(), y.max()], 
                        origin='lower', aspect='auto', vmin=vmins[j],
                        vmax = vmaxs[j])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.10)
            cbar = fig.colorbar(h, cax=cax)
            # cbar.ax.tick_params(labelsize=15)
            if (j == 2 or j==4) and i==5:
                # ax.plot([-1, 1], [0.0, 0.0], color='white')
                pass

            ax.set_xlabel(r'$x/\mathrm{m}$')
            ax.set_ylabel(r'$y/\mathrm{m}$')
            # font size doubled
            ax.set_title('${}(x,y,t)$ prediction'.format(plot_name[j]))
            ax.set_aspect(1)
        fig.suptitle("time = {:.2f} s".format(time))
        fig.tight_layout()
        fig.savefig("img/heatmap_{:02d}.png".format(i))