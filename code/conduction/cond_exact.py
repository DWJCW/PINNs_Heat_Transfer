import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.linalg import lu

qs = 1
L = 1
W = 1
Nx = 50
Ny = 50

dx = L/Nx
dy = W/Ny
dt = 1e-2
k = 1
alpha = 1
T_i = 0
T_1 = 0

def create_equ(Nx, Ny, dx, dy, dt, qs, k, alpha):
    # Create Equations
    equ_cube = np.zeros([(Nx+1)*(Ny+1), Nx+1, Ny+1])
    b = np.zeros((Nx+1)*(Ny+1))
    p = 0
    for i in range(1, Nx):
        for j in range(1, Ny):
            equ_cube[p, i, j] = 1/(alpha*dt) + 2/np.square(dx) + 2/np.square(dy)
            equ_cube[p, i, j+1] = -1/np.square(dx)
            equ_cube[p, i, j-1] = -1/np.square(dx)
            equ_cube[p, i+1, j] = -1/np.square(dy)
            equ_cube[p, i-1, j] = -1/np.square(dy)
            b[p] = 0
            p += 1

    # Equs for the top except for the first and the last
    i = 0

    for j in range(1, Nx):
        equ_cube[p, i, j] = 1/(2*alpha*dt) + 1/np.square(dx) + 1/np.square(dy)
        equ_cube[p, i, j+1] = -0.5/np.square(dx)
        equ_cube[p, i, j-1] = -0.5/np.square(dx)
        equ_cube[p, i+1, j] = -1/np.square(dy)
        b[p] = qs/(k*dy)
        p += 1

    # Equs for the rest
    for i in range(0, Ny+1):
        equ_cube[p, i, 0] = 1
        b[p] = T_1
        p += 1
        equ_cube[p, i, Nx] = 1
        b[p] = T_1
        p += 1    

    for j in range(1, Nx):
        equ_cube[p, Ny, j] = 1
        b[p] = T_1
        p += 1

    equ = np.array([equ_cube[i].reshape((Nx+1)*(Ny+1)) for i in range((Nx+1)*(Ny+1))])
    return equ, b

def solve(equ, b, iter_num, T):
    # Solve
    # b_t = T.reshape((Nx+1)*(Ny+1))
    equ_inv = np.linalg.inv(equ)

    T0 = T
    for i in range(iter_num):
        temp = [T0[i, j]/(alpha*dt) for i in range(1, Ny) for j in range(1, Nx)]
        temp += [T0[0, j]/(2*alpha*dt) for j in range(1, Nx)]
        temp += [0]*(Nx*2+Ny+1)   
        b_now = b + temp
        T0 = np.matmul(equ_inv, b_now)
        T0 = T0.reshape((Nx+1, Ny+1))
    return T0

def plot_temp(T0, L, W, Nx, Ny, fig, ax):
    # Plot
    xs = np.linspace(0, L, Nx+1)
    ys = np.linspace(0, W, Ny+1)
    [xs, ys] = np.meshgrid(xs, ys)
    ax.invert_yaxis()

    cs = ax.contourf(xs, ys, T0, 100, cmap=cm.jet)
    colorbar = fig.colorbar(cs)
    colorbar.set_ticks(np.linspace(0, 1, 11))
    colorbar.set_label(r'$T/\mathrm{^\circ C}$')
    ax.set_aspect(1)

def check_stable(T0, T1, dt):
    dT = np.abs(T0-T1)
    dT_dt = dT/dt
    return np.mean(dT_dt)


if __name__=="__main__":
    equ, b = create_equ(Nx, Ny, dx, dy, dt, qs, k, alpha)
    T0 = np.ones([Nx+1, Ny+1])*T_i

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    Ts = [T0]

    for current_time in np.arange(dt, 1+dt, dt):
        T1 = solve(equ, b, 1, T0)
        T0 = T1
        Ts.append(T0)
        print(current_time)
    
    plot_temp(T0, L, W, Nx, Ny, fig, ax)
    plt.show()
    np.save("Ts", Ts)