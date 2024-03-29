# Comparison of three high-level programming languages for numerical computing

## Objective

Compare the syntax and ease-of-use of these three popular programming languages on basic numerical computation tasks:

- Python (Python Software Foundation)
- MATLAB (Mathworks)
- Julia (https://julialang.org)

### Other things not considered here

- R (www.r-project.org) a high-level language for statistical computing
- Capabilities of these languages in other applications
- A comparison of the performance of these languages
  - See here for example: https://julialang.org/benchmarks/ 
- Maturity of the language, available packages and support community


## Language Comparison – Examples

The following sections contain code examples in each language for the following features:

+ [Function definition](#function-definition)
+ [One-line functions](#one-line-functions)
+ [Variable assignment](#variable-assignment)
+ [Standard output](#standard-output)
+ [Formatted output](#formatted-output)
+ [Definite loop](#definite-loop)
+ [Indefinite loop](#indefinite-loop)
+ [Lists](#lists)
+ [Dictionaries](#dictionaries)
+ [Iterating over collections](#iterating-over-collections)
+ [Array literals](#array-literals)
+ [Array dimensions](#array-dimensions)
+ [Indexing arrays](#indexing-arrays)
+ [Slicing arrays](#slicing-arrays)
+ [Array concatenation](#array-concatenation)
+ [Array broadcasting](#array-broadcasting)
+ [Linear algebra](#linear-algebra)
+ [Symbolic Math](#symbolic-math)


### Function definition

Python

```Python
def power(x, a):
    return x**a
```

Julia
```Julia
function power(x, a)
  x^a
end

# Assignment form
power2(x, a) = x^a
```

MATLAB
```Matlab
function y = power(x, a)
  y = x^a;
end

% Note: Functions must go at end of the
% file or in a separate file
```


### Anonymous functions

Python

```Python
# Lambda function
power = lambda x, a: x**a
```

Julia
```Julia
# Anonymous function
power = (x, a) ->  x^a
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
# Since Python 3.4
print("{:5.2f}".format(x))

# Since Python 3.6
print(f"{x:5.2f}")

# 'Old style'
print("%5.2f" % x)

# Output:
#  1.41
#  1.41
#  1.41
```

Julia
```Julia
using Printf

@printf("%5.2f\n", x)

# Output:
#  1.41
```

MATLAB
```Matlab
fprintf("%5.2f\n", x)

% Output:
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
i = 1
while i <= 5
  println(i)
  global i += 1
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


### Lists

Python
```Python
symbols = ['H', 'He', 'Li']
print(symbols)

# Output:
# ['H', 'He', 'Li']
```

Julia
```Julia
symbols = ["H", "He", "Li"]  # Array
println(symbols)

# Output
# ["H", "He", "Li"]
```

MATLAB

```Matlab
symbols = {'H','He','Li'}  % Cell array

% Output:
% 
% symbols =
% 
%   1×3 cell array
% 
%     {'H'}    {'He'}    {'Li'}
% 
```


### Dictionaries

Python
```Python
elements = {
    'H': 1,
    'He': 2, 
    'Li': 3
}
print(elements)

# Output:
# {'H': 1, 'He': 2, 'Li': 3}
```

Julia
```Julia
elements = Dict(
    "H" => 1,
    "He" => 2, 
    "Li" => 3
)
println(elements)

# Output
# Dict("Li"=>3,"He"=>2,"H"=>1)
```

MATLAB

```Matlab
symbols = {'H','He','Li'};  % numeric or cell arrays only
values = [1 2 3];
elements = containers.Map(symbols,values)

% Output:
% 
% elements = 
% 
%   Map with properties:
% 
%         Count: 3
%       KeyType: char
%     ValueType: double
% 
```

### Iterating over collections

Python
```Python
for symbol, n in elements.items():
    print(f"{symbol}: {n}")

# Output:
# H: 1
# He: 2
# Li: 3
```

Julia
```Julia
for (symbol, n) in elements
  @printf("%s: %d\n", symbol, n)
end

# Output:
# H: 1
# He: 2
# Li: 3
```

MATLAB
```Matlab
for symbol = keys(elements)
    n = elements(symbol{1});
    fprintf("%s: %d\n", symbol{1}, n)
end

% Output:
% H: 1
% He: 2
% Li: 3
```


### Array literals

Python

```Python
import numpy as np

# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# 3d array (3x2x3)
Z = np.ones((3, 2, 3))

print(x)
print(A)
print(Z)

# Output:
# [1 2 3]
# [[1 2]
#  [3 4]
#  [5 6]]
# array([[[1., 1., 1.],
#         [1., 1., 1.]],
# 
#        [[1., 1., 1.],
#         [1., 1., 1.]],
# 
#        [[1., 1., 1.],
#         [1., 1., 1.]]])
# 
```

Julia

```Julia
# Vector (1d)
x = [1, 2, 3]
println(x)

# Matrix (2d)
A = [1 2; 3 4; 5 6]
println(A)

# 3d array (3x2x3)
Z = ones(3, 2, 3)

println(x)
println(A)
println(Z)

# Output:
# [1, 2, 3]
# [1 2; 3 4; 5 6]
# [1.0 1.0; 1.0 1.0; 1.0 1.0]
# 
# [1.0 1.0; 1.0 1.0; 1.0 1.0]
# 
# [1.0 1.0; 1.0 1.0; 1.0 1.0]
# 
```

MATLAB
```Matlab
% Row vector (2d)
x = [1 2 3]  % or x = [1, 2, 3]

% Matrix (2d)
A = [1 2; 3 4; 5 6]

% 3d array (3x2x3)
Z = ones(3, 2, 3)

% Output
% 
% x =
% 
%      1     2     3
% 
% 
% A =
% 
%      1     2
%      3     4
%      5     6
% 
% 
% Z(:,:,1) =
% 
%      1     1
%      1     1
%      1     1
% 
% 
% Z(:,:,2) =
% 
%      1     1
%      1     1
%      1     1
% 
% 
% Z(:,:,3) =
% 
%      1     1
%      1     1
%      1     1
% 
```

### Array dimensions

Python

```Python
import numpy as np

# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# 3d array (3x2x3)
Z = np.ones((3, 2, 3))

print(x.shape)
print(A.shape)
print(Z.shape)

# Output:
# (3,)
# (3, 2)
# (3, 2, 3)
```

Julia
```Julia
# Vector (1d)
x = [1, 2, 3]

# Matrix (2d)
A = [1 2; 3 4; 5 6]

# 3d array (3x2x3)
Z = ones(3, 2, 3)

println(size(x))
println(size(A))
println(size(Z))

# Output:
# (3,)
# (2, 3)
# (3, 2, 3)
```

MATLAB
```Matlab
% Row vector (2d)
x = [1 2 3]  % or x = [1, 2, 3]

% Matrix (2d)
A = [1 2; 3 4; 5 6]

% 3d array (3x2x3)
Z = ones(3, 2, 3)

size(x)
size(A)
size(Z)

% Output
% 
% ans =
% 
%      1     3
% 
% 
% ans =
% 
%      3     2
% 
% 
% ans =
% 
%      3     2     3
% 
```


### Indexing arrays

Python

```Python
# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print(x[1])
print(A[1,1])
print(A[1])

# Output:
# 2
# 4
# [3 4]
```

Julia
```Julia
# Vector (1d)
x = [1, 2, 3]

# Matrix (2d)
A = [1 2; 3 4; 5 6]

println(x[2])
println(A[2,2])
println(A[2])

# Output:
# 2
# 4
# 3
```

MATLAB
```Matlab
% Row vector (2d)
x = [1 2 3]  % or x = [1, 2, 3]

% Matrix (2d)
A = [1 2; 3 4; 5 6]

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
%      4
% 
% 
% ans =
% 
%      3
% 
```


### Slicing arrays

Python

```Python
# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print(x[1:])
print(A[:,1])

# Output:
# [2 3]
# [2 4 6]
```

Julia
```Julia
# Vector (1d)
x = [1, 2, 3]

# Matrix (2d)
A = [1 2; 3 4; 5 6]

println(x[2:end])
println(A[:,2])

# Output:
# [2, 3]
# [2, 4, 6]
```

MATLAB
```Matlab
% Row vector (2d)
x = [1 2 3]  % or x = [1, 2, 3]

% Matrix (2d)
A = [1 2; 3 4; 5 6]

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
%      4
%      6
% 
```


### Array concatenation

Python

```Python
# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Concatenation of vector and 2-d array
C = np.hstack([x.reshape(-1, 1), A])
# or
C = np.column_stack([x, A])
print(C)

# Output:
# [[1 1 2]
#  [2 3 4]
#  [3 5 6]]
```

Julia
```Julia
# Vector (1d)
x = [1, 2, 3]
println(x)

# Matrix (2d)
A = [1 2; 3 4; 5 6]
println(A)

# Concatenation of vector and 2-d array
C = [x A]
println(C)

# Output
# [1 1 2; 2 3 4; 3 5 6]
```

MATLAB
```Matlab
% Row vector (2d)
x = [1 2 3]  % or x = [1, 2, 3]

% Matrix (2d)
A = [1 2; 3 4; 5 6]

% Concatenation of vector and 2-d array
C = [x' A]

% Output
% 
% C =
% 
%      1     1     2
%      2     3     4
%      3     5     6
% 
```


### Array broadcasting

Python

```Python
# Vector (1d)
x = np.array([1, 2, 3])

# Matrix (2d)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Array broadcasting
print(1 - A)
print(A + x.reshape(-1, 1))

# Output:
# [[ 0 -1]
#  [-2 -3]
#  [-4 -5]]
# [[2 3]
#  [5 6]
#  [8 9]]
```

Julia
```Julia
# Row vector (2d)
x = [1 2 3]

# Matrix (2d)
A = [1 2; 3 4; 5 6]

println(1 .- A)
println(A .+ x)

# Output:
# [0 -1; -2 -3; -4 -5]
# [2 3; 5 6; 8 9]
```

MATLAB
```Matlab
% Row vector (2d)
x = [1 2 3]  % or x = [1, 2, 3]

% Matrix (2d)
A = [1 2; 3 4; 5 6]

1-A
A+x'

% Output
% 
% ans =
% 
%      0    -1
%     -2    -3
%     -4    -5
% 
% 
% ans =
% 
%      2     3
%      5     6
%      8     9
% 
```


### Linear algebra

Python

```Python
import numpy as np

A = np.array([[0.8, 0], [0, 1]])
K = np.array([[0.942], [1.074]])
C = np.array([[0.2, 1]])

# Matrix inverse
# Since Python 3.5
X = np.linalg.inv(np.eye(2) - A + K @ C)
# Prior to 3.5
X = np.linalg.inv(np.eye(2) - A + K.dot(C))

print(X)

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

# Matrix inverse
X = (I - A + K*C)^-1

println(X)

# Output:
# [5.000000000000001 -4.385474860335195; -1.0000000000000002 1.808193668528864]
```

MATLAB
```Matlab
A = [0.8 0; 0 1]; 
K = [0.942; 1.074]; 
C = [0.2 1];

% Matrix inverse
X = (eye(2) - A + K*C)^-1

% Output
% 
% ans =
% 
%     5.0000   -4.3855
%    -1.0000    1.8082
% 
```

### Symbolic Math

Python

```Python
from sympy import symbols, solve, diff
a, b, c, x = symbols('a b c x')
y = a*x**2 + b*x + c
dydx = diff(y, x)
x_sol = solve(dydx, x)
y_sol = y.subs(x, x_sol[0])
print(dydx)
print(x_sol)
print(y_sol)

# Output:
# 2*a*x + b
# [-b/(2*a)]
# c - b**2/(4*a)
```

Julia
```Julia
y = :(a*x^2 + b*x + c)  # Symbolic expressions are built in
using Reduce
dydx = Algebra.df(y,:x)
x_sol = Algebra.solve(dydx,:x)
y_sol = Algebra.sub(x_sol,y)
println(dydx)
println(x_sol)
println(y_sol)

# Output:
# 2 * a * x + b
# (:(x = -b / (2a)),)
# (4 * a * c - b ^ 2) / (4a)
```

MATLAB
```Matlab
syms a b c x
y = a*x^2 + b*x + c;
dydx = diff(y,x)
x_sol = solve(dydx,x)
y_sol = subs(y,x,x_sol)

% Output
% 
% dydx =
%  
% b + 2*a*x
%  
%  
% x_min =
%  
% -b/(2*a)
%  
%  
% y_min =
%  
% - b^2/(4*a) + c
%  
```
