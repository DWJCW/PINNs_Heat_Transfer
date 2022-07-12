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


alpha = 1

SAVE_PATH = "cond.pt"


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

    def netT(self, x, y, t):
        Temp = self.dnn(torch.cat([x, y, t], dim=1))
        return Temp

    def netF(self, x, y, t):
        Temp = self.netT(x, y, t)
        Tt = self.grad(Temp, t)
        Tx = self.grad(Temp, x)
        Ty = self.grad(Temp, y)
        Txx = self.grad(Tx, x)
        Tyy = self.grad(Ty, y)
        f = Tt - alpha*(Txx+Tyy)
        return f

    def printLoss(self, loss, loss_f, loss_bcs):
        print("-"*15*5)
        info1 = "{:<15}{:<15}".format("Iter:", self.iter)
        info2 = "{:<15}{:<15}".format(
            "loss", "loss_f")
        info3 = "{:<15.5e}{:<15.5e}{:<15.5e}".format(
            loss.item(), loss_f.item(), torch.sum(loss_bcs).item())
        print(info1)
        print(info2)
        print(info3)
        info4 = "{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}".format( 
            "", "init", "bc1", "bc2", "bc3", "bc4"
        )
        print(info4)
        for i in range(1):
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
        f_pred= self.netF(x, y, t)
        loss_f = torch.mean((f_pred)**2)
        loss_bcs = torch.zeros(len(self.X_bcs), 1)
        for i in range(len(self.X_bcs)):
            X_bc = self.X_bcs[i]
            F_bc = self.F_bcs[i]
            x, y, t = self.getXYT(X_bc)
            Temp = self.netT(x, y, t)
            j = 0
            for key, value in F_bc.items():
                if key == "T":
                    bc_pred = Temp
                elif key == "Tx":
                    bc_pred = self.grad(Temp, x)
                elif key == "Ty":
                    bc_pred = self.grad(Temp, y)
                loss_bcs[i, j]  = torch.mean((bc_pred - value)**2)
                j += 1
        loss_bcs[0, 0] = loss_bcs[0, 0]*100
        loss = loss_f + torch.sum(loss_bcs)

        # loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            self.printLoss(loss, loss_f, loss_bcs)
        return loss

    def train(self):
        self.dnn.train()
        loss = self.loss_func()
        for epoch in range(10):
            loss_queue = [1., 0.]
            print("Epoch: {}".format(epoch))
            print("#"*100)
            while True:
                loss = self.loss_func()
                loss_queue.insert(0, loss.item())

                # Backward and optimize
                self.optimizer_Adam.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer_Adam.step()
                if len(loss_queue)>100:
                    loss_start = loss_queue.pop()
                    if np.abs(loss.item()-loss_start)/np.mean(loss_queue)<1e-4:
                        break
            self.data.createMesh()
            self.createData(self.data)
            self.iter = 0
            torch.save(self.dnn, SAVE_PATH)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

        self.dnn.eval()
        Temp = self.netT(x, y, t)
        return Temp


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
        self.x_lim = np.array([0, 1])
        self.y_lim = np.array([0, 1])
        self.t_lim = np.array([0, 1])
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
        Temp = np.ones_like(self.X_init[:, 0:1])*0.
        self.F_init = {
            "T": Temp,
        }

        # Left B.C., x = -15, U=(1, 0), p' = 0
        X, Y, T = np.meshgrid(x[0:1], y, t)
        self.X_bc1 = self.mesh2List(X, Y, T)
        Temp = np.ones_like(self.X_bc1[:, 0:1])*0.
        self.F_bc1 = {
            "T": Temp,
        }

        # Top B.C., y = 8
        X, Y, T = np.meshgrid(x, y[-1::], t)
        self.X_bc2 = self.mesh2List(X, Y, T)
        Tempy = np.ones_like(self.X_bc2[:, 0:1])*1.
        self.F_bc2 = {
            "Ty": Tempy,
        }

        # Right B.C., x = 25
        X, Y, T = np.meshgrid(x[-1::], y, t)
        self.X_bc3 = self.mesh2List(X, Y, T)
        Temp = np.ones_like(self.X_bc3[:, 0:1])*0.
        self.F_bc3 = {
            "T": Temp,
        }

        # Bottom B.C., x = 25
        X, Y, T = np.meshgrid(x, y[0:1], t)
        self.X_bc4 = self.mesh2List(X, Y, T)
        Temp = np.ones_like(self.X_bc4[:, 0:1])*0.
        self.F_bc4 = {
            "T": Temp,
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
    N_b = 1000
    N_F = 1000
    N_x = 1000
    N_y = 1000
    N_t = 1000
    data = Data(N_b, N_F, N_x, N_y, N_t)
    layers = [3, 10, 20, 40, 40, 40, 20, 10, 5, 1]
    # layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = PINNs(data, layers)
    if 1:
        model.dnn = torch.load(SAVE_PATH)
        model.optimizer_Adam = torch.optim.Adam(
            model.dnn.parameters(),
            lr=1e-6,
        )
    model.train()
    torch.save(model.dnn, SAVE_PATH)