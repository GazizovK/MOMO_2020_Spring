import oracles
from plot_trajectory_2d import plot_levels
from plot_trajectory_2d import plot_trajectory
import optimization
import numpy as np
import matplotlib.pyplot as plt

oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
[x_star, msg, history] = optimization.gradient_descent(oracle, np.array([3.0, 1.5]), trace=True)
plt.figure()
plot_levels(oracle.func)
plot_trajectory(oracle.func, history['x'])
plt.plot()
