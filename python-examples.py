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
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(x)
print(A)

# Nd array (2x3x2)
Z = np.ones((2, 3, 2))
print(Z)

# Array dimensions
print(x.shape)
print(A.shape)

# Indexing arrays
print(x[1])
print(A[1,1])
print(A[1])

# Slicing arrays
print(x[1:])
print(A[:,1])

# Array broadcasting
print(1 - A)
print(A + x)

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

# Speed test results
# Test 1 - Single function call: 2.42 µs ± 72
# Test 2 - Single function calls with for loop: 37.4 ms ± 465 µs
# Test 3 - Vectorized function calls: 58.2 µs ± 381 ns

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
# %timeit dy = make_function_calls(lorenz, y, n, sigma=sigma, beta=beta, rho=rho)
# %timeit dy = lorenz(y, sigma, beta, rho)
