import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ns import *

if __name__ == "__main__":
    exact = pd.read_csv("exact2.csv")
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
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    model = PINNs(data, layers)
    model.dnn = torch.load("ns1.pt")

    u_pred, v_pred, p_pred = model.predict(X_pred)
    u_pred = u_pred.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    # p_pred = p_pred.detach().cpu().numpy()
    Umag_pred = np.sqrt(np.square(u_pred)+np.square(v_pred))
    plt.plot(x, Umag, label="exact")
    plt.plot(x_pred+1, Umag_pred, "-.", label="PINNs")
    plt.legend()
    plt.show()