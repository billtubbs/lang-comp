# Comparison of high-level programming languages
# This is Python (requires version 3.6+)

# Function
def power(x, a):
    return x**a

# Lambda function
power = lambda x, a: x**a

# Variable assignment
x = power(2, 0.5)

# Standard output
print(x)

# Formatted output
print("{:5.2f}".format(x))  # Since 3.4
print(f"{x:5.2f}")  # Since 3.6
print("%5.2f" % x)  # 'Old style'

# Definite loop
for i in range(5):
    print(i)

# Indefinite loop
i = 0
while i < 5:
    print(i)
    i += 1

# List
symbols = ['H', 'He', 'Li']
print(symbols)

# Dictionary
elements = {
    'H': 1,
    'He': 2,
    'Li': 3
}
print(elements)

# Iterating over collections
for symbol, n in elements.items():
    print(f"{symbol}: {n}")

# Array literals
import numpy as np

# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(x)
print(A)

# 3d array (3x2x3)
Z = np.ones((3, 2, 3))
print(Z)

# Array dimensions
print(x.shape)
print(A.shape)
print(Z.shape)

# Indexing arrays
print(x[1])
print(A[1,1])
print(A[1])

# Slicing arrays
print(x[1:])
print(A[:,1])

# Concatenation of vector and 2-d array
C = np.hstack([x.reshape(-1, 1), A])
# or
C = np.column_stack([x, A])
print(C)

# Array broadcasting
print(1 - A)
print(A + x.reshape(-1, 1))

# Linear Algebra
A = np.array([[0.8, 0], [0, 1]])
K = np.array([[0.942],[1.074]])
C = np.array([[0.2, 1]])

# Matrix inverse
# Since Python 3.5
X = np.linalg.inv(np.eye(2) - A + K @ C)
# Prior to 3.5
X = np.linalg.inv(np.eye(2) - A + K.dot(C))

print(X)

# Symbolic math
from sympy import symbols, solve, diff
a, b, c, x = symbols('a b c x')
y = a*x**2 + b*x + c
dydx = diff(y, x)
x_sol = solve(dydx, x)
y_sol = y.subs(x, x_sol[0])
print(dydx)
print(x_sol)
print(y_sol)

# Vectorizable function
def lorenz(y, sigma, beta, rho):
    return (sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2])

# Parameter values
beta = 8 / 3
sigma = 10
rho = 28

# Test1 - Single function call
y0 = (-8, 8, 27)
assert lorenz(y0, sigma, beta, rho) == (160, -16, -136)

# Test 2 - Single function calls with for loop
def make_function_calls(f, y, n, *args, **kwargs):
    dy = np.empty_like(y)
    for i in range(n):
        dy[:, i] = f(y[:, i], *args, **kwargs)
    return dy

y = np.repeat(np.array((-8, 8, 27)).reshape(-1,1), 10, axis=1)
dy = make_function_calls(lorenz, y, 10, sigma=sigma, beta=beta, rho=rho)
assert dy.shape == y.shape
assert np.array_equiv(dy.T, (160, -16, -136))

# Test 3 - vectorized function call
y = np.repeat(np.array((-8, 8, 27)).reshape(-1,1), 10, axis=1)
dy = lorenz(y, sigma, beta, rho)
dy = np.vstack(dy)
assert dy.shape == y.shape
assert np.array_equiv(dy.T, (160, -16, -136))

# Test 4 - Numpy apply method (slowest, not vectorized)
dy = np.apply_along_axis(lorenz, 0, y, sigma=sigma, beta=beta, rho=rho)
assert dy.shape == y.shape
assert np.array_equiv(dy.T, (160, -16, -136))

# Test 5 - Numpy vectorized function
vlorenz = np.vectorize(lorenz, excluded={"sigma", "beta", "rho"})

# Speed test results
# Test 1 - Single function call: 504 ns ± 6.08 ns
# Test 2 - Single function calls with for loop: 37.2 ms ± 302 µs
# Test 3 - Vectorized function calls: 59.0 µs ± 381 ns
# Test 4 - Numpy apply method 54.7 ms ± 511 µs

# Timing code (use in IPython or Jupyter notebook):
# n = 10000
# y = np.random.randn(3, n)
# y0 = y[:, 0]
# def make_function_calls(f, y, n, *args, **kwargs):
#     dy = np.empty_like(y)
#     for i in range(n):
#         dy[:, i] = f(y[:, i], *args, **kwargs)
#     return dy
# %timeit dy = lorenz(y0, sigma, beta, rho)
# %timeit dy = make_function_calls(lorenz, y, n, sigma=sigma,
#                                  beta=beta, rho=rho)
# %timeit dy = lorenz(y, sigma, beta, rho)
# %timeit dy = np.apply_along_axis(lorenz, 0, y, sigma=sigma,
#                                  beta=beta, rho=rho)


# Lorenz system with forcing
def lorenz(y, u, sigma=10, beta=8/3, rho=28):
    return (sigma * (y[1] - y[0]) + u,
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2])
