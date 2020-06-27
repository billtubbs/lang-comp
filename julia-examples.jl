# Comparison of high-level programming languages
# This is Julia (version 1.3.1)

# Function
function power(x, a)
  return x^a
end

# Assignment form
power(x, a) = x^a

# Variable assignment
x = power(2, 0.5)

# Standard output
println(x)

# Formatted output
using Printf
@printf("%5.2f\n", x)

# Definite loop
for i in 1:5
  println(i)
end

# Indefinite loop
i = 1
while i <= 5
  println(i)
  global i += 1
end

# List
symbols = ["H", "He", "Li"]  # Array
println(symbols)

# Dictionary
elements = Dict(
    "H" => 1,
    "He" => 2, 
    "Li" => 3
)
println(elements)

# Iterating over collections
for (symbol, n) in elements
  @printf("%s: %d\n", symbol, n)
end

# Array literals

# Vector (1d)
x = [1, 2, 3]
println(x)

# Matrix (2d)
A = [1 2 3; 4 5 6]
println(A)

# Nd array (2x3x2)
Z = ones(2, 3, 2)
println(Z)

# Array dimensions
println(size(x))
println(size(A))

# Indexing arrays
println(x[2])
println(A[2,2])
println(A[2])

# Slicing arrays
println(x[2:end])
println(A[:,2])

# Row vector (2d)
x = [1 2 3]

# Array broadcasting
println(1 .- A)
println(A .+ x)

# Linear algebra
using LinearAlgebra
A = [0.8 0; 0 1];
K = [0.942; 1.074];
C = [0.2 1];
X = (I - A + K*C)^-1
println(X)

