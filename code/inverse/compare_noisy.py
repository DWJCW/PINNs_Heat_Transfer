import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

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
    i = 1
    path = "nn0{}/loss".format(i)
    path = "noisy_nn/loss"

    # Read data from loss file and make it become numpy array
    df = pd.read_csv(path, delim_whitespace=True)
    print(df.columns)
    n = df['noisy'].to_numpy()
    alpha = df['alpha'].to_numpy()
    loss = df['loss'].to_numpy()
    time = df['time'].to_numpy()
    pred_alpha = df['pred_alpha'].to_numpy()

    # Reshape them to be a matrix
    N = n.reshape(5, 5)
    Alpha = alpha.reshape(5, 5)
    Loss = loss.reshape(5, 5)
    Time = time.reshape(5, 5)
    Pred_alpha = pred_alpha.reshape(5, 5)
    Alpha_error = np.abs(Alpha-Pred_alpha)/Alpha

    fig = plt.figure(figsize=[6, 2])

    # Plot loss
    ax = fig.add_subplot(1, 3, 1)
    img = ax.imshow(Loss.T, interpolation='nearest', cmap=cm.jet,
                    norm=LogNorm(
                        vmin=10**(int(np.log10(np.min(Loss)))), vmax=1),
                    extent=[N.min(), N.max(), alpha.min(), alpha.max()],
                    origin='lower', aspect='auto')
    ax.set_aspect((np.max(N) - np.min(N))/(np.max(alpha) - np.min(alpha)))
    cbar = fig.colorbar(img, fraction=0.046, pad=0.04)
    ax.set_title("loss")
    # ax.set_title("Loss")
    ax.set_xlabel(r"noise")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_yticks([0.2, 0.6, 1.0])

    # Plot time
    ax = fig.add_subplot(1, 3, 2)
    img = ax.imshow(Time.T, interpolation='nearest', cmap=cm.jet,
                    extent=[N.min(), N.max(), alpha.min(), alpha.max()],
                    origin='lower', aspect='auto')
    ax.set_aspect((np.max(N) - np.min(N))/(np.max(alpha) - np.min(alpha)))
    cbar = fig.colorbar(img, fraction=0.046, pad=0.04)
    ax.set_title("time / s")
    # ax.set_title("Loss")
    ax.set_xlabel(r"noise")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_yticks([0.2, 0.6, 1.0])

    # Plot error alpha
    ax = fig.add_subplot(1, 3, 3)
    print(Alpha_error)
    print(Pred_alpha)
    print(Alpha)
    img = ax.imshow(Alpha_error.T, interpolation='nearest', cmap=cm.jet,
                    norm=LogNorm(
                        vmin=10**(int(np.log10(np.min(Alpha_error)))),
                        vmax=1),
                    extent=[N.min(), N.max(), alpha.min(), alpha.max()],
                    origin='lower', aspect='auto')
    ax.set_aspect((np.max(N) - np.min(N))/(np.max(alpha) - np.min(alpha)))
    cbar = fig.colorbar(img, fraction=0.046, pad=0.04)
    ax.set_title(r"$e_\alpha$")
    # ax.set_title("Loss")
    ax.set_xlabel(r"noise")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_yticks([0.2, 0.6, 1.0])

    # save and plot
    fig.tight_layout(pad=0.0)
    fig.savefig("img/noisy.png")
    # plt.show()
