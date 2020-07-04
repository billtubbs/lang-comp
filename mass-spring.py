from itertools import accumulate 
import numpy as np
import matplotlib.pyplot as plt


def mass_spring(x, u, dt=0.05, m=1, c=2, k=1): 
    """Mass-spring-damper system
    x  :: state variables (x1, x2) where
         x1 = position (m)
         x2 = velocity (m/s)
    u  :: input force (N)
    dt :: timestep (s)
    m  :: mass (kg)
    c  :: damping coefficient (Ns/m)
    k  :: spring constant (N/m)
    """
    dxdt = (x[1], (-k*x[0] - c*x[1] + u) / m)
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


# Unit tests
# Mass spring
assert mass_spring((0.0, 0.0), 0.0) == (0.0, 0.0)
dt, m, c, k = (0.1, 1, 0.2, 1)
x1 = mass_spring((0.0, 0.0), 0.2, dt, m=m, c=c, k=k)
assert np.allclose(x1, (0.0, 0.02))
x1 = mass_spring((0.2, 0.0), 0.0, dt, m=m, c=c, k=k)
assert np.allclose(x1, (0.2, -0.02))
x1 = mass_spring((0.0, 0.2), 0.0, dt, m=m, c=c, k=k)
assert np.allclose(x1, (0.02, 0.196))

# Simulation with forcing
x0 = (0.0, 0.0)
nT = 200  # number of timesteps
t_sim = np.linspace(0, 10, nT + 1)
u_sim = np.zeros(nT)
u_sim[0:25] = 1  # Add force for short period

# Solve recurrence equation
x_sol = list(accumulate(u_sim, mass_spring, initial=x0))
assert len(x_sol) == nT + 1
plot_trajectory(t_sim, x_sol, u_sim, title="Mass-Spring Damper")
