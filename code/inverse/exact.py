import numpy as np


def getExact(xs: float, ts: float, alpha: float):
    """Return the exact temperature distribution

    Parameters
    ----------
    xs : float
        spatial coordinate, it can also be a np.ndarray
    ts : float
        temporal coordinate, it can also be a np.ndarray
    alpha : float
        thermal diffusivity
    """
    def ln(n): return np.pi/2 + n*np.pi
    def Cn(n): return 4*np.sin(ln(n))/(2*ln(n)+np.sin(2*ln(n)))

    def theta(eta, tau):
        sum = 0.
        i = 0
        while True:
            l_n = ln(i)
            new_term = Cn(i)*np.cos(l_n*eta)*np.exp(-np.square(l_n)*tau)
            sum += new_term
            i += 1
            if np.max(np.abs(new_term)) <= 1e-12:
                break
        return sum

    L = 1.
    etas = xs/L
    taus = alpha * ts/L**2
    thetas = theta(etas, taus)
    return thetas
