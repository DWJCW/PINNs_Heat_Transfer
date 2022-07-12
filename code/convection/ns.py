import torch
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
from pyDOE import lhs
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib import cm

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

nu = 0.01


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1),
             torch.nn.Linear(layers[-2], layers[-1]))
        )
        layer_dict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.layers(x)
        return out


class PINNs():
    def __init__(self, data, layers):
        self.data = data
        self.createData(self.data)
        self.layers = layers

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(),
            lr=1e-4,
        )

        self.iter = 0

    def createData(self, data):
        self.lb = data.lb
        self.ub = data.ub
        for i, X_bc in enumerate(data.X_bcs):
            data.X_bcs[i] = torch.tensor(
                X_bc, requires_grad=True).float().to(device)
        for F_bc in data.F_bcs:
            for key, value in F_bc.items():
                F_bc[key] = torch.tensor(
                    value, requires_grad=True).float().to(device)
        self.X_bcs = data.X_bcs
        self.F_bcs = data.F_bcs
        self.X_gov = torch.tensor(
            data.X_gov, requires_grad=True).float().to(device)

    def getXYT(self, input):
        X = input[:, 0:1]
        Y = input[:, 1:2]
        T = input[:, 2:3]
        return X, Y, T

    def grad(self, y, x):
        y_x = torch.autograd.grad( 
            y, x, grad_outputs=torch.ones_like(y),
            retain_graph=True, create_graph=True
        )[0]
        return y_x

    def netPsiP(self, x, y, t):
        psi_p = self.dnn(torch.cat([x, y, t], dim=1))
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        return psi, p

    def netU(self, x, y, psi):
        u = self.grad(psi, y)
        v = -self.grad(psi, x)
        return u, v

    def netF(self, x, y, t):
        psi, p = self.netPsiP(x, y, t)
        u, v = self.netU(x, y, psi)
        ut = self.grad(u, t)
        vt = self.grad(v, t)
        ux = self.grad(u, x)
        vx = self.grad(v, x)
        uy = self.grad(u, y)
        vy = self.grad(v, y)
        px = self.grad(p, x)
        py = self.grad(p, y)
        uxx = self.grad(ux, x)
        vxx = self.grad(vx, x)
        uyy = self.grad(uy, y)
        vyy = self.grad(vy, y)
        fu = ut + (u*ux+v*uy) + px - nu*(uxx + uyy)
        fv = vt + (u*vx+v*vy) + py - nu*(vxx + vyy)
        fc = ux + vy
        return fu, fv, fc

    def printLoss(self, loss, loss_fu, loss_fv, loss_fc, loss_bcs):
        print("-"*15*5)
        info1 = "{:<15}{:<15}".format("Iter:", self.iter)
        info2 = "{:<15}{:<15}{:<15}{:<15}{:<15}".format(
            "loss", "loss_fu", "loss_fv", "loss_fc", "loss_bc")
        info3 = "{:<15.5e}{:<15.5e}{:<15.5e}{:<15.5e}{:<15.5e}".format(
            loss.item(), loss_fu.item(), loss_fv.item(), loss_fc.item(),
            torch.sum(loss_bcs).item())
        print(info1)
        print(info2)
        print(info3)
        info4 = "{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}".format( 
            "", "init", "bc1", "bc2", "bc3", "bc4"
        )
        print(info4)
        for i in range(3):
            info5 = \
                "{:<15}{:<15.5e}{:<15.5e}{:<15.5e}{:<15.5e}{:<15.5e}".\
                    format("Type{}".format(i+1),
                    loss_bcs[0, i].item(), 
                    loss_bcs[1, i].item(), 
                    loss_bcs[2, i].item(), 
                    loss_bcs[3, i].item(), 
                    loss_bcs[4, i].item(), 
                    )
            print(info5)

    def loss_func(self):
        # self.optimizer.zero_grad()
        x, y, t = self.getXYT(self.X_gov)
        fu_pred, fv_pred, fc_pred = self.netF(x, y, t)
        loss_fu = torch.mean((fu_pred)**2)
        loss_fv = torch.mean((fv_pred)**2)
        loss_fc = torch.mean((fc_pred)**2)
        loss_bcs = torch.zeros(len(self.X_bcs), 3)
        for i in range(len(self.X_bcs)):
            X_bc = self.X_bcs[i]
            F_bc = self.F_bcs[i]
            x, y, t = self.getXYT(X_bc)
            psi, p = self.netPsiP(x, y, t)
            u, v = self.netU(x, y, psi)
            j = 0
            for key, value in F_bc.items():
                if key == "U":
                    bc_pred = u
                elif key == "V":
                    bc_pred = v
                elif key == "Ux":
                    bc_pred = self.grad(u, x)
                elif key == "Uy":
                    bc_pred = self.grad(u, y)
                elif key == "Vx":
                    bc_pred = self.grad(v, x)
                elif key == "Vy":
                    bc_pred = self.grad(v, y)
                elif key == "P":
                    bc_pred = p
                elif key == "Px":
                    bc_pred = self.grad(p, x)
                elif key == "Py":
                    bc_pred = self.grad(p, y)
                elif key == "Pn":
                    px = self.grad(p, x)
                    py = self.grad(p, y)
                    bc_pred = px*x + py*y
                loss_bcs[i, j]  = torch.mean((bc_pred - value)**2)
                j += 1
        loss = loss_fu + loss_fv + loss_fc + torch.sum(loss_bcs)

        # loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            self.printLoss(loss, loss_fu, loss_fv, loss_fc, loss_bcs)
        return loss

    def train(self):
        self.dnn.train()
        loss = self.loss_func()
        for epoch in range(10):
            print("Epoch: {}".format(epoch))
            print("#"*100)
            while self.iter<=5000:
                loss = self.loss_func()

                # Backward and optimize
                self.optimizer_Adam.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer_Adam.step()
            self.data.createMesh()
            self.createData(self.data)
            self.iter = 0
            torch.save(self.dnn, "ns1.pt")

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

        self.dnn.eval()
        psi, p = self.netPsiP(x, y, t)
        u, v = self.netU(x, y, psi)
        return u, v, p


class Data:
    def __init__(self, N_b:int, N_F:int, N_x:int, N_y:int, N_t:int):
        """
        Generate data for PINNs

        Parameters
        ----------
        N_b : int
            boundary node number
        N_F : int
            internal node number
        N_x : int
            mesh for x
        N_y : int
            mesh for y
        N_t : int
            mesh for t
        """
        self.N_b = N_b
        self.N_F = N_F
        self.N_x = N_x
        self.N_y = N_y
        self.N_t = N_t

        # set basic parameter for the mesh
        self.x_lim = np.array([-1, 1])
        self.y_lim = np.array([-0.25, 0.25])
        self.t_lim = np.array([0, 5.])
        self.lb = np.array([self.x_lim[0], self.y_lim[0], self.t_lim[0]])
        self.ub = np.array([self.x_lim[1], self.y_lim[1], self.t_lim[1]])

        self.createMesh()
    
    def mesh2List(self, X:np.ndarray, Y:np.ndarray, T:np.ndarray):
        """
        make mesh matrix into a list, for example:
        X = [[1, 2],
             [3, 4]]
        Y = [[4, 3],
             [2, 1]]
        T = [[-1, -2],
             [-3, -4]]
        Then, it will return:
        [[1 4 -1]
         [2 3 -2]
         [3 2 -3]
         [4 1 -4]]

        Parameters
        ----------
        X : np.ndarray
            _description_
        Y : np.ndarray
            _description_
        T : np.ndarray
            _description_
        """
        return np.hstack(
            [ 
                X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]
            ]
        )
    
    def plotData(self, X, Y):
        plt.plot(
            X, 
            Y, 
            'kx', label = 'Data (%d points)' % (X.shape[0]), 
            markersize = 4,  # marker size doubled
            clip_on = False,
            alpha=1.0
        )
    
    def dataSample(self, input, output, N):
        # Make a sampling among the input array and output dictionary
        idx = np.random.choice(input.shape[0], N, replace=False)
        input = input[idx, :]
        for key, value in output.items():
            idx = np.random.choice(value.shape[0], N, replace=False)
            output[key] = value[idx, :]
        return input, output
    
    def createMesh(self):
        x = np.linspace(self.x_lim[0], self.x_lim[1], self.N_x)
        y = np.linspace(self.y_lim[0], self.y_lim[1], self.N_y)
        t = np.linspace(self.t_lim[0], self.t_lim[1], self.N_t)

        # I.C., t = 0
        X, Y, T = np.meshgrid(x, y, t[0:1])
        self.X_init = self.mesh2List(X, Y, T)
        U = np.ones_like(self.X_init[:, 0:1])*0.
        V = np.ones_like(self.X_init[:, 0:1])*0.
        P = np.ones_like(self.X_init[:, 0:1])*0.
        self.F_init = {
            "U": U,
            "V": V,
            "P": P
        }

        # Left B.C., x = -15, U=(1, 0), p' = 0
        X, Y, T = np.meshgrid(x[0:1], y, t)
        self.X_bc1 = self.mesh2List(X, Y, T)
        U = np.ones_like(self.X_bc1[:, 0:1])*1.
        V = np.ones_like(self.X_bc1[:, 0:1])*0.
        Px = np.ones_like(self.X_bc1[:, 0:1])*0.
        self.F_bc1 = {
            "U": U,
            "V": V,
            "Px": Px,
        }

        # Top B.C., y = 8
        X, Y, T = np.meshgrid(x, y[-1::], t)
        self.X_bc2 = self.mesh2List(X, Y, T)
        U = np.ones_like(self.X_bc2[:, 0:1])*0.
        V = np.ones_like(self.X_bc2[:, 0:1])*0.
        Py = np.ones_like(self.X_bc2[:, 0:1])*0.
        self.F_bc2 = {
            "U": U,
            "V": V,
            "Py": Py,
        }

        # Right B.C., x = 25
        X, Y, T = np.meshgrid(x[-1::], y, t)
        self.X_bc3 = self.mesh2List(X, Y, T)
        Ux = np.ones_like(self.X_bc3[:, 0:1])*0.
        Vx = np.ones_like(self.X_bc3[:, 0:1])*0.
        P = np.ones_like(self.X_bc3[:, 0:1])*0.
        self.F_bc3 = {
            "Ux": Ux,
            "Vx": Vx,
            "P": P,
        }

        # Bottom B.C., x = 25
        X, Y, T = np.meshgrid(x, y[0:1], t)
        self.X_bc4 = self.mesh2List(X, Y, T)
        U = np.ones_like(self.X_bc4[:, 0:1])*0.
        V = np.ones_like(self.X_bc4[:, 0:1])*0.
        Py = np.ones_like(self.X_bc4[:, 0:1])*0.
        self.F_bc4 = {
            "U": U,
            "V": V,
            "Py": Py,
        }

        self.X_init, self.F_init = self.dataSample(
            self.X_init, self.F_init, self.N_b)
        self.X_bc1, self.F_bc1 = self.dataSample(
            self.X_bc1, self.F_bc1, self.N_b)
        self.X_bc2, self.F_bc2 = self.dataSample(
            self.X_bc2, self.F_bc2, self.N_b)
        self.X_bc3, self.F_bc3 = self.dataSample(
            self.X_bc3, self.F_bc3, self.N_b)
        self.X_bc4, self.F_bc4 = self.dataSample(
            self.X_bc4, self.F_bc4, self.N_b)
        
        self.X_bcs = [
            self.X_init, self.X_bc1, self.X_bc2, self.X_bc3, self.X_bc4]
        self.F_bcs = [
            self.F_init, self.F_bc1, self.F_bc2, self.F_bc3, self.F_bc4]
        
        self.X_gov = self.lb + (self.ub-self.lb)*lhs(3, self.N_F)
        self.X_gov = np.vstack(
            [self.X_gov, self.X_init, self.X_bc1, self.X_bc2, self.X_bc3, 
            self.X_bc4]
        )


if __name__=="__main__":
    N_b = 2000
    N_F = 10000
    N_x = 1000
    N_y = 1000
    N_t = 1000
    data = Data(N_b, N_F, N_x, N_y, N_t)
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    model = PINNs(data, layers)
    if 1:
        model.dnn = torch.load("ns1.pt")
        model.optimizer_Adam = torch.optim.Adam(
            model.dnn.parameters(),
            lr=1e-6,
        )
    model.train()
    torch.save(model.dnn, "ns1.pt")