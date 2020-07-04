# Demonstrates various methods to solve a recurrence
# equation in Python

from itertools import accumulate
from functools import reduce
import numpy as np
#import matplotlib.pyplot as plt


# Example: discrete approximation of Duffing's oscillator
def duffing(x, u, dt=0.05, delta=0.2):
    """Duffing's oscillator"""
    dxdt = (x[1], x[0] - delta*x[1] - x[0]**3 + u)
    return x[0] + dt*dxdt[0], x[1] + dt*dxdt[1]


def plot_trajectory(t, x, u, title=None):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(t, [x[0] for x in x_sol], label='$x_1(t)$')
    axes[0].plot(t, [x[1] for x in x_sol], label='$x_2(t)$')
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title(title)
    axes[1].step(t[:-1], u, label='$u(t)$')
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xlabel('t')
    plt.show()


# Simulation with forcing
x0 = (0.0, 0.0)  # Initial state
nT = 200  # number of timesteps
t_sim = np.linspace(0, 10, nT + 1)
u_sim = np.zeros(nT)  # Input signal
u_sim[0:25] = 1  # Add force for short period

# Solve recurrence equation
# (requires Python 3.8+)
x_sol = list(accumulate(u_sim, duffing, initial=x0))
assert len(x_sol) == nT + 1

# Builtin reduce can be used to get final value
x_final = reduce(duffing, u_sim, x0)
assert np.allclose(x_final, x_sol[-1])

# Uncomment to make plot:
#plot_trajectory(t_sim, x_sol, u_sim, title="Duffing's Oscillator")

# Equivalent computation with a for loop
def solve_recurrence(x0, f, u):
    x, x_sol = x0, [x0]
    for k in range(nT):
        x = duffing(x, u[k])
        x_sol.append(x)
    return x_sol

x_sol2 = solve_recurrence(x0, duffing, u_sim)
assert all([x1 == x2 for (x1, x2) in zip(x_sol2, x_sol)])

# Equivalent computation with a list comprehension
# (requires Python 3.8+)
x_sol3 = [x := x0] + [(x := duffing(x, u)) for u in u_sim]
assert all([x1 == x2 for (x1, x2) in zip(x_sol3, x_sol)])


# Speed test results
# x_sol = list(accumulate(u_sim, duffing, initial=x0))
# 426 µs ± 10.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# x_sol2 = solve_recurrence(x0, duffing, u_sim)
# 471 µs ± 6.55 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# x_sol3 = [x := x0] + [(x := duffing(x, u)) for u in u_sim]
# 440 µs ± 8.16 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# %timeit x_final = reduce(duffing, u_sim, x0)      
# 418 µs ± 5.19 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
