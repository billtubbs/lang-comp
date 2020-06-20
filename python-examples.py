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
A = np.array([[0.8, 0], 
              [0, 1]])
K = np.array([[0.942],
              [1.074]])
C = np.array([0.2, 1])
print(np.linalg.inv(np.eye(2) 
      - A + K*C))

