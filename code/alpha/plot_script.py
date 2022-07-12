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

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out

# the physics-guided neural network
class PINNs():
    def __init__(self, X_u, u, X_f, layers):
        # data
        self.x_u = torch.tensor(
            X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(
            X_u[:, 1:2], requires_grad=True).float().to(device)
        self.alpha_u = torch.tensor(
            X_u[:, 2:3], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(
            X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(
            X_f[:, 1:2], requires_grad=True).float().to(device)
        self.alpha_f = torch.tensor(
            X_f[:, 2:3], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        
        self.layers = layers

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )

        self.iter = 0
    
    def net_u(self, x, t, alpha):
        u = self.dnn(torch.cat([x, t, alpha], dim=1))
        return u
    
    def net_f(self, x, t, alpha):
        u = self.net_u(x, t, alpha)

        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t - alpha * u_xx
        return f
    
    def loss_func(self):
        self.optimizer.zero_grad()
        
        u_pred = self.net_u(self.x_u, self.t_u, self.alpha_u)
        f_pred = self.net_f(self.x_f, self.t_f, self.alpha_f)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        
        loss = loss_u + loss_f
        
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss

    def train(self):
        self.dnn.train()

        # Backward and optimize
        self.optimizer.step(self.loss_func)

            
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        alpha = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t, alpha)
        f = self.net_f(x, t, alpha)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


T_i = 1.0
T_t = 0.0
T_b = 0.0

N_u = 1000
N_f = 10000
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]

N_t = 100
N_x = 100
N_a = 100
t = np.linspace(0, 1, N_t)
x = np.linspace(-1, 1, N_x)
alpha = np.linspace(0.0, 1, N_a)

X, T, ALPHA = np.meshgrid(x, t, alpha)
X_star = np.hstack(
    (X.flatten()[:,None], T.flatten()[:,None], ALPHA.flatten()[:,None]))

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

 # Left boundary, t=0
xx1 = np.hstack(
    (X[0:1,:,:].flatten()[:,None], 
    T[0:1,:,:].flatten()[:,None], 
    ALPHA[0:1,:,:].flatten()[:,None]))
uu1 = np.ones_like(xx1) * T_i
# Lower boundary, x=-1
xx2 = np.hstack(
    (X[:,0:1,:].flatten()[:,None], 
    T[:,0:1,:].flatten()[:,None], 
    ALPHA[:,0:1,:].flatten()[:,None]))
uu2 = np.ones_like(xx2) * T_b
# Upper boundary, x=1
xx3 = np.hstack(
    (X[:,-1:,:].flatten()[:,None], 
    T[:,-1:,:].flatten()[:,None], 
    ALPHA[:,-1:,:].flatten()[:,None]))
uu3 = np.ones_like(xx3) * T_t

X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = lb + (ub-lb)*lhs(3, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]


dnn = torch.load('alpha.pt')
model = PINNs(X_u_train, u_train, X_f_train, layers)
model.dnn = dnn.to(device)


alpha_values = [0.2, 0.4, 0.6, 0.8, 1.0]

for alpha_value in alpha_values:
    # Prediction from PINNs
    N_t = 200
    N_x = 200
    N_a = 1
    t = np.linspace(0, 1, N_t)
    x = np.linspace(-1, 1, N_x)
    alpha = np.linspace(alpha_value, alpha_value, N_a)

    X, T, ALPHA = np.meshgrid(x, t, alpha)
    X_pred = np.hstack(
        (X.flatten()[:,None], T.flatten()[:,None], ALPHA.flatten()[:,None]))

    u_pred, f_pred = model.predict(X_pred)
    U_pred = u_pred.reshape(N_x, N_t)

    ln = lambda n: np.pi/2 + n*np.pi
    Cn = lambda n: 4*np.sin(ln(n))/(2*ln(n)+np.sin(2*ln(n)))
    def theta(eta, tau, N):
        sum = 0.
        for i in range(N):
            l_n = ln(i)
            sum += Cn(i)*np.cos(l_n*eta)*np.exp(-np.square(l_n)*tau)
        return sum

    ts = np.linspace(0, 1, N_t)
    xs = np.linspace(0, 1, int(N_x/2))
    alphas = np.linspace(alpha_value, alpha_value, N_a)
    [Xs, Ts] = np.meshgrid(xs, ts)
    L = 1.

    N = 2000
    etas = xs/L
    taus = alpha_value * ts/L**2
    Etas, Taus = np.meshgrid(etas, taus)
    Thetas = theta(Etas, Taus, N)

    U_exact = np.hstack((Thetas[:, -1::-1], Thetas))
    x = np.linspace(-1, 1, N_x)
    t = np.linspace(0, 1, N_t)
    alpha = np.linspace(alpha_value, alpha_value, N_a)

    X, T, ALPHA = np.meshgrid(x, t, alpha)
    X_exact = np.hstack(
        (X.flatten()[:,None], T.flatten()[:,None], ALPHA.flatten()[:,None]))



    """ The aesthetic setting has changed. """

    ####### Row 0: u(t,x) ##################    

    fig = plt.figure(figsize=(6, 3))
    # fig.suptitle("$\\alpha={:.2f}$".format(alpha_value), fontsize = 24)

    ###### Prediction #####################
    ax = fig.add_subplot(211)

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap=cm.jet, 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    # cbar.ax.tick_params(labelsize=15) 

    '''ax.plot(
        X_u_train[:,1], 
        X_u_train[:,0], 
        'kx', label = 'Data (%d points)' % (u_train.shape[0]), 
        markersize = 4,  # marker size doubled
        clip_on = False,
        alpha=1.0
    )
    '''

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[int(N_t/4)]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[int(N_t*2/4)]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[int(N_t*3/4)]*np.ones((2,1)), line, 'w-', linewidth = 1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$T(t,x)$ prediction') # font size doubled

    ####### Exact ###############

    ax = fig.add_subplot(212)

    h = ax.imshow(U_exact.T, interpolation='nearest', cmap=cm.jet, 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[int(N_t/4)]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[int(N_t*2/4)]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[int(N_t*3/4)]*np.ones((2,1)), line, 'w-', linewidth = 1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.9, -0.05), 
        ncol=5, 
        frameon=False, 
    )
    ax.set_title('$T(t,x)$ exact') # font size doubled

    fig.tight_layout(pad=-0.2)
    fig.savefig("img/heat_map{}.png".format(alpha_value))

    ####### Row 1: u(t,x) slices ################## 

    """ The aesthetic setting has changed. """

    fig = plt.figure(figsize=(6, 3))
    # fig.suptitle("$\\alpha={:.2f}$".format(alpha_value), fontsize = 20)
    ax = fig.add_subplot(111)

    # gs1 = gridspec.GridSpec(1, 3)
    # gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(221)
    ax.plot(x,U_exact[1,:], 'b-', linewidth = 2, label = 'Exact')    
    ax.plot(x,U_pred[1,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$T(t,x)$')    
    ax.set_title('$t = 0.01$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([0,1.1])
    ax.set_aspect(1)

    ax = plt.subplot(222)
    ax.plot(x,U_exact[int(N_t/4),:], 'b-', linewidth = 2, label = 'Exact')    
    ax.plot(x,U_pred[int(N_t/4),:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$T(t,x)$')    
    ax.set_title('$t = 0.25$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([0,1.1])
    ax.set_aspect(1)

    ax = plt.subplot(223)
    ax.plot(x,U_exact[int(N_t*2/4),:], 'b-', linewidth = 2, label = 'Exact')    
    ax.plot(x,U_pred[int(N_t*2/4),:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$T(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([0,1.1])
    ax.set_title('$t = 0.50$')
    ax.set_aspect(1)

    ax = plt.subplot(224)
    ax.plot(x,U_exact[int(N_t*3/4),:], 'b-', linewidth = 2, label = 'Exact')     
    ax.plot(x,U_pred[int(N_t*3/4),:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$T(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([0,1.1])    
    ax.set_title('$t = 0.75$')
    ax.set_aspect(1)
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(-0.1, -0.2), 
        ncol=5, 
        frameon=False, 
    )
    fig.tight_layout(pad=0.1)
    fig.savefig("img/line{}.png".format(alpha_value))