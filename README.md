# lang-comp
Comparison of three high-level programming languages for numerical computing.

## Objective

Compare the syntax and ease-of-use of three popular programming languages on basic numerical computation tasks:

- Python (Python Software Foundation)
- MATLAB (Mathworks)
- Julia (https://julialang.org)


## Things not considered here

- R (www.r-project.org) a high-level language for for statistical computing
- Capabilities of these languages in other applications
- A comparison of the performance of these languages
 - See here for example: https://julialang.org/benchmarks/ 
- Maturity of the language, available packages and support community


## Language Comparison – Examples

### Function definition

Python

```Python
def power(x, a):
    return x**a
```

Julia
```Julia
function power(x, a)
  return x^a
end
```

MATLAB
```Matlab
function y = power(x, a)
  y = x^a;
end

% Note: Functions must go at end of the
% file or in a separate file
```


### One-line functions

Python

```Python
# Lambda function
power = lambda x, a: x**a
```

Julia
```Julia
# Assignment form
power(x, a) = x^a
```

MATLAB
```Matlab
% Anonymous function
power = @(x, a) x^a
```

### Variable assignment

Python

```Python
x = power(2, 0.5)
```

Julia
```Julia
x = power(2, 0.5)
```

MATLAB
```Matlab
x = power(2, 0.5);
```


### Standard output

Python

```Python
print(x)

# Output:
# 1.4142135623730951
```

Julia
```Julia
println(x)

# Output:
# 1.4142135623
```

MATLAB
```Matlab
disp(x)

% Output:
%   1.4142

% Or omit the line terminator
x = power(2, 0.5)

% Output:
% 
% x =
% 
%     1.4142
% 
```


### Formatted output

Python

```Python
# Since 3.4
print("{:5.2f}".format(x))

#  1.41

# Since 3.6
print(f"{x:5.2f}")

#  1.41

# 'Old style'
print("%5.2f" % x)

#  1.41
```

Julia
```Julia
using Printf

@printf("%5.2f\n", x)

#  1.41
```

MATLAB
```Matlab
fprintf("%5.2f\n", x)

%  1.41
```


### Definite loop

Python

```Python
for i in range(5):
    print(i)

# Output:
# 0
# 1
# 2
# 3
# 4
```

Julia
```Julia
for i in 1:5
  println(i)
end

# Output:
# 1
# 2
# 3
# 4
# 5
```

MATLAB
```Matlab
for i=1:5
  disp(i)
end

% Output:
%      1
% 
%      2
% 
%      3
% 
%      4
% 
%      5
% 
```


### Indefinite loop

Python

```Python
i = 0
while i < 5:
    print(i)
    i += 1

# Output:
# 0
# 1
# 2
# 3
# 4
```

Julia
```Julia
i = 0
while i < 5:
    print(i)
    i += 1

# Output:
# 1
# 2
# 3
# 4
# 5
```

MATLAB
```Matlab
i = 1;
while i <= 5
  disp(i)
  i = i + 1;
end

% Output:
%      1
% 
%      2
% 
%      3
% 
%      4
% 
%      5
% 
```


### Array literals

Python

```Python
import numpy as np

# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2, 3], 
              [4, 5, 6]])

# Nd array (2x3x2)
Z = np.ones((2, 3, 2))
print(Z)

print(x)
print(A)
print(Z)

# Output:
# [1 2 3]
# [[1 2 3]
#  [4 5 6]]
# [[[1. 1.]
#   [1. 1.]
#   [1. 1.]]
# 
#  [[1. 1.]
#   [1. 1.]
#   [1. 1.]]]
```

Julia
```Julia
# Vector (1d)
x = [1, 2, 3]

# Matrix (2d)
A = [1 2 3; 4 5 6]

# Nd array (3x3x2)
Z = ones(2, 3, 2)

println(x)
println(A)
println(Z)

# Output:
# [1, 2, 3]
# [1 2 3; 4 5 6]
# [1.0 1.0 1.0; 1.0 1.0 1.0]
# 
# [1.0 1.0 1.0; 1.0 1.0 1.0]
```

MATLAB
```Matlab
% Column vector (2d)
x = [1, 2, 3]

% Matrix (2d)
A = [1 2 3; 4 5 6]

% Output
% 
% x =
% 
%      1     2     3
% 
% 
% A =
% 
%      1     2     3
%      4     5     6
% 
% 
% Z(:,:,1) =
% 
%      1     1     1
%      1     1     1
% 
% 
% Z(:,:,2) =
% 
%      1     1     1
%      1     1     1
% 
```

### Array dimensions

Python

```Python
import numpy as np

# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2, 3], 
              [4, 5, 6]])

print(x.shape)
print(A.shape)

# Output:
# (3,)
# (2, 3)
```

Julia
```Julia
# Vector (1d)
x = [1, 2, 3]

# Matrix (2d)
A = [1 2 3; 4 5 6]

println(size(x))
println(size(A))

# Output:
# (3,)
# (2, 3)
```

MATLAB
```Matlab
% Column vector (2d)
x = [1, 2, 3];

% Matrix (2d)
A = [1 2 3; 4 5 6];

size(x)
size(A)

% Output
% 
% ans =
% 
%      1     3
% 
% 
% ans =
% 
%      2     3
% 
```


### Indexing arrays

Python

```Python
# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2, 3], 
              [4, 5, 6]])

print(x[1])
print(A[1,1])
print(A[1])

# Output:
# 2
# 5
# [4 5 6]
```

Julia
```Julia
# Vector (1d)
x = [1, 2, 3]

# Matrix (2d)
A = [1 2 3; 4 5 6]

println(x[2])
println(A[2,2])
println(A[2])

# Output:
# 2
# 5
# 4
```

MATLAB
```Matlab
% Column vector (2d)
x = [1, 2, 3];

% Matrix (2d)
A = [1 2 3; 4 5 6];

x(2)
A(2,2)
A(2)


% Output
% 
% ans =
% 
%      2
% 
% 
% ans =
% 
%      5
% 
% 
% ans =
% 
%      4
% 
```


### Slicing arrays

Python

```Python
# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2, 3], 
              [4, 5, 6]])

print(x[1:])
print(A[:,1])

# Output:
# [2 3]
# [2 5]
```

Julia
```Julia
# Vector (1d)
x = [1, 2, 3]

# Matrix (2d)
A = [1 2 3; 4 5 6]

println(x[2:end])
println(A[:,2])

# Output:
# [2, 3]
# [2, 5]
```

MATLAB
```Matlab
% Column vector (2d)
x = [1, 2, 3];

% Matrix (2d)
A = [1 2 3; 4 5 6];

x(2:end)
A(:,2)

% Output
% 
% ans =
% 
%      2     3
% 
% 
% ans =
% 
%      2
%      5
% 
```


### Array broadcasting

Python

```Python
# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2, 3], 
              [4, 5, 6]])

print(1 - A)
print(A + x)

# Output:
# [[ 0 -1 -2]
#  [-3 -4 -5]]
# [[2 4 6]
#  [5 7 9]]
```

Julia
```Julia
# Row vector (2d)
x = [1 2 3]

# Matrix (2d)
A = [1 2 3; 4 5 6]

println(1 .- A)
println(A .+ x)

# Output:
# [0 -1 -2; -3 -4 -5]
# [2 4 6; 5 7 9]
```

MATLAB
```Matlab
% Row vector (2d)
x = [1 2 3];

% Matrix (2d)
A = [1 2 3; 4 5 6];

1 - A
A + x

% Output
% 
% ans =
% 
%      0    -1    -2
%     -3    -4    -5
% 
% 
% ans =
% 
%      2     4     6
%      5     7     9
% 
```


### Linear algebra

Python

```Python
import numpy as np

A = np.array([[0.8, 0], 
              [0, 1]])
K = np.array([[0.942],
              [1.074]])
C = np.array([0.2, 1])

% Matrix inverse
print(np.linalg.inv(np.eye(2) 
      - A + K*C))

# Output:
# [[ 5.         -4.38547486]
#  [-1.          1.80819367]]
```

Julia
```Julia
using LinearAlgebra

A = [0.8 0; 0 1];
K = [0.942; 1.074];
C = [0.2 1];

% Matrix inverse
println((I - A + K*C)^-1)

# Output:
# [5.000000000000001 -4.385474860335195; -1.0000000000000002 1.808193668528864]
```

MATLAB
```Matlab
A = [0.8 0; 0 1]; 
K = [0.942; 1.074]; 
C = [0.2 1];

% Matrix inverse
(eye(2) - A + K*C)^-1

% Output
% 
% ans =
% 
%     5.0000   -4.3855
%    -1.0000    1.8082
% 
```
