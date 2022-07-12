import torch
import numpy as np
from collections import OrderedDict
from pyDOE import lhs
from matplotlib import pyplot as plt
from exact import getExact
import time


# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

SAVE_PATH = "inverse.pt"


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

        alpha = torch.tensor(np.random.rand())
        self.alpha = torch.autograd.Variable(alpha, requires_grad=True)

        self.optimizer_Adam = torch.optim.Adam(
            [{"params": self.dnn.parameters()}, {"params": self.alpha}],
            lr=1e-4,
        )

        self.iter = 0

    def createData(self, data):
        self.lb = data.lb
        self.ub = data.ub
        self.X_gov = torch.tensor(
            data.X_gov, requires_grad=True).float().to(device)
        self.X_known = torch.tensor(
            data.X_known, requires_grad=True).float().to(device)
        self.F_known = torch.tensor(
            data.F_known, requires_grad=True).float().to(device)

    def getXT(self, input):
        X = input[:, 0:1]
        T = input[:, 1:2]
        return X, T

    def grad(self, y, x):
        y_x = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            retain_graph=True, create_graph=True
        )[0]
        return y_x

    def netT(self, x, t):
        Temp = self.dnn(torch.cat([x, t], dim=1))
        return Temp

    def netF(self, x, t):
        Temp = self.netT(x, t)
        Tt = self.grad(Temp, t)
        Tx = self.grad(Temp, x)
        Txx = self.grad(Tx, x)
        f = Tt - self.alpha*(Txx)
        return f

    def printLoss(self, loss, loss_f, loss_known):
        print("-"*15*5)
        info1 = "{:<15}{:<15}".format("Iter:", self.iter)
        info2 = "{:<15}{:<15}{:<15}{:<15}".format(
            "loss", "loss_f", "loss_known", "pred_alpha")
        info3 = "{:<15.5e}{:<15.5e}{:<15.5e}{:<15.5e}".format(
            loss.item(), loss_f.item(), loss_known.item(), self.alpha.item())
        print(info1)
        print(info2)
        print(info3)

    def loss_func(self):
        # self.optimizer.zero_grad()
        x, t = self.getXT(self.X_gov)
        f_pred = self.netF(x, t)
        loss_f = torch.mean((f_pred)**2)

        x, t = self.getXT(self.X_known)
        T_pred = self.netT(x, t)
        loss_known = torch.mean((T_pred - self.F_known)**2)
        loss = loss_f + loss_known

        # loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            self.printLoss(loss, loss_f, loss_known)
        return loss

    def train(self):
        self.dnn.train()
        loss = self.loss_func()
        for epoch in range(10):
            loss_queue = [1., 0.]  # used to check if converge
            print("Epoch: {}".format(epoch))
            print("#"*100)
            while True:
                loss = self.loss_func()
                loss_queue.insert(0, loss.item())  # used to check if converge

                # Backward and optimize
                self.optimizer_Adam.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer_Adam.step()
                if len(loss_queue) > 100:
                    loss_start = loss_queue.pop()
                    if np.abs(loss.item()-loss_start)/np.mean(loss_queue) < \
                            1e-4:
                        break
            self.data.createMesh()
            self.createData(self.data)
            self.iter = 0
            torch.save(self.dnn, SAVE_PATH)
        return loss.item()

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        Temp = self.netT(x, t)
        return Temp


class Data:
    def __init__(self, N_F: int, N_known: int, test_alpha: float):
        """_summary_

        Parameters
        ----------
        N_F : int
            number for gov node
        N_known : int
            number for given node
        test_alpha : float
            tested alpha
        """
        self.N_F = N_F
        self.N_known = N_known
        # set basic parameter for the mesh
        self.x_lim = np.array([-1, 1])
        self.t_lim = np.array([0, 1])
        self.lb = np.array([self.x_lim[0], self.t_lim[0]])
        self.ub = np.array([self.x_lim[1], self.t_lim[1]])
        self.test_alpha = test_alpha

        self.X_known = self.lb + (self.ub-self.lb)*lhs(2, self.N_known)
        X = self.X_known[:, 0:1]
        T = self.X_known[:, 1:2]
        self.F_known = getExact(X, T, self.test_alpha)

        self.createMesh()

    def plotData(self, X, T):
        plt.plot(
            X,
            T,
            'kx', label='Data (%d points)' % (X.shape[0]),
            markersize=4,  # marker size doubled
            clip_on=False,
            alpha=1.0
        )
        plt.show()

    def dataSample(self, input, output, N):
        # Make a sampling among the input array and output dictionary
        idx = np.random.choice(input.shape[0], N, replace=False)
        input = input[idx, :]
        for key, value in output.items():
            idx = np.random.choice(value.shape[0], N, replace=False)
            output[key] = value[idx, :]
        return input, output

    def createMesh(self):
        self.X_gov = self.lb + (self.ub-self.lb)*lhs(2, self.N_F)
        self.X_gov = np.vstack([self.X_gov, self.X_known])


def train_N():
    N_konwns = [20, 40, 60, 80, 100]
    alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
    with open("nn/loss", "a") as f:
        info = "{:<15}{:<15}{:<15}{:<15}{:<15}\n".format(
            "N", "alpha", "loss", "time", "pred_alpha")
        # print(info)
        f.write(info)
    for N_known in N_konwns:
        for test_alpha in alphas:
            start = time.time()
            N_F = 1000
            data = Data(N_F, N_known, test_alpha)
            layers = [2, 10, 20, 40, 40, 40, 20, 10, 5, 1]
            model = PINNs(data, layers)
            loss = model.train()
            path = "nn/N_{}_alpha_{}.pt".format(N_known, int(test_alpha*10))
            end = time.time()
            with open("nn/loss", "a") as f:
                info = "{:<15}{:<15.8e}{:<15.8e}{:<15.8e}{:<15.8e}\n".format(
                        N_known, test_alpha, loss, end-start,
                        model.alpha.item())
                print(info)
                f.write(info)
            torch.save(model.dnn, path)


if __name__ == "__main__":
    train_N()
    if 0:
        N_F = 1000
        N_known = 10
        test_alpha = 0.5
        data = Data(N_F, N_known, test_alpha)
        layers = [2, 10, 20, 40, 40, 40, 20, 10, 5, 1]
        model = PINNs(data, layers)
        if 0:
            model.dnn = torch.load(SAVE_PATH)
            model.optimizer_Adam = torch.optim.Adam(
                model.dnn.parameters(),
                lr=1e-6,
            )
        loss = model.train()
        torch.save(model.dnn, SAVE_PATH)
