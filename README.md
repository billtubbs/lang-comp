# lang-comp
Comparison of three high-level programming languages for numerical computing.

## Objective

Compare the syntax and ease-of-use of three popular programming languages on basic numerical computation tasks:

- Python (Python Software Foundation)
- MATLAB (Mathworks)
- Julia (https://julialang.org)


## Other things not considered here

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
