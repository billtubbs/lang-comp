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
    return x\*\*a

# Lambda function
power = lambda x, a: x**a
```

Julia
```Julia
function power(x, a)
    return x^a
end

# Assignment form
power(x, a) = x^a
```

MATLAB
```Matlab
function y = power(x, a)
  y = x^a;
end


% Anonymous function
power = @(x, a) x^a

% Note: Functions must go at end of the
% file or in a separate file
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

# 1.4142135623730951
```

Julia
```Julia
println(x)

# 1.4142135623
```

MATLAB
```Matlab
disp(x)

%   1.4142

% Or omit line terminator
x = power(2, 0.5)

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
```

Julia
```Julia
```

MATLAB
```Matlab
```
