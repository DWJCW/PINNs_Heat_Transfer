from exact import getExact
import numpy as np
from matplotlib import pyplot as plt


xs = np.linspace(-1, 1, 100)
plt.plot(xs, getExact(xs, 0.001, 0.1))
plt.show()
