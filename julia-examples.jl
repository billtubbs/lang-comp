# Comparison of high-level programming languages
# This is Julia (version 1.3.1)

# Function
function power(x, a)
  x^a
end
three_squared = power(3, 2)

# Anonymous function
power2 = (x, a) ->  x^a
@assert(power2(3, 2) == power(3, 2))

# Assignment form
power3(x, a) = x^a
@assert(power3(3, 2) == power(3, 2))

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
elements = ["H", "He", "Li"]  # Array
println(elements)

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
A = [1 2; 3 4; 5 6]
println(A)

# 3d array (3x2x3)
Z = ones(3, 2, 3)
println(Z)

# Array dimensions
println(size(x))
println(size(A))
println(size(Z))

# Indexing arrays
println(x[2])
println(A[2,2])
println(A[2])

# Slicing arrays
println(x[2:end])
println(A[:,2])

# Concatenation of vector and 2-d array
C = [x A]
println(C)

# Array broadcasting
println(1 .- A)
println(A .+ x)

# Linear algebra
using LinearAlgebra
A = [0.8 0; 0 1]
K = [0.942; 1.074]
C = [0.2 1]
X = (I - A + K*C)^-1
println(X)

# Symbolic math
y = :(a*x^2 + b*x + c)  # Symbolic expressions are built in
using Reduce
dydx = Algebra.df(y,:x)
x_sol = Algebra.solve(dydx,:x)
y_sol = Algebra.sub(x_sol,y)
println(dydx)
println(x_sol)
println(y_sol)

# Non-vectorized method
function lorenz(y1, y2, y3, sigma::Float64, 
                beta::Float64, rho::Float64)
  return (sigma * (y2 - y1), y1 * (rho - y3) - y2, 
          y1 * y2 - beta * y3)
end
lorenz(y::Tuple{Float64,Float64,Float64}, 
       sigma::Float64, beta::Float64, 
       rho::Float64) = (
         lorenz(y[1], y[2], y[3], sigma, beta, rho)
       )

# Vectorized method 1
# function lorenz(y1::Array, y2::Array, y3::Array, 
#                 sigma::Float64, beta::Float64, 
#                 rho::Float64)
#   dy1 = sigma.*(y2.-y1)
#   dy2 = y1.*(rho.-y3).-y2
#   dy3 = y1.*y2.-beta.*y3
#   return [dy1 dy2 dy3]'
# end
# lorenz(y::Array, sigma::Float64, beta::Float64, 
#        rho::Float64) = (
#          lorenz(y[1,:], y[2,:], y[3,:], sigma, beta, rho)
#        )
# Vectorized method 2
# function lorenz(y::Array, sigma::Float64, beta::Float64, 
#   rho::Float64)
# dy = similar(y)
# dy[1,:] = sigma.*(y[2,:].-y[1,:])
# dy[2,:] = y[1,:].*(rho.-y[3,:]).-y[2,:]
# dy[3,:] = y[1,:].*y[2,:].-beta.*y[3,:]
# return dy
# end
# Vectorized method 3
function lorenz(
  y::Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1}}, 
  sigma::Float64, beta::Float64, rho::Float64
  )
  return (sigma.*(y[2].-y[1]), y[1].*(rho.-y[3]).-y[2], 
          y[1].*y[2].-beta.*y[3])
end
# This is not faster than method 3 above:
function lorenz(y::Array{Float64,2}, sigma::Float64, beta::Float64, 
                rho::Float64)
  return (sigma.*(y[2,:].-y[1,:]), y[1,:].*(rho.-y[3,:]).-y[2,:], 
          y[1,:].*y[2,:].-beta.*y[3,:])
end


# Parameter values
beta = 8 / 3
sigma = 10.0
rho = 28.0
y0 = (-8.0, 8.0, 27.0)
dy = lorenz(y0, sigma, beta, rho)
println(dy)
@assert dy == (160.0, -16.0, -136.0)

# Test - vectorized method 2
# y = repeat([-8.0, 8.0, 27.0], 1, 10)
# dy = lorenz(y, sigma, beta, rho)
# @assert size(dy) == size(y)
# @assert dy == repeat([160.0,  -16.0, -136.0], 1, 10)

# Test - vectorized method 3
y = (repeat([-8.0],10), repeat([8.0],10), repeat([27.0],10))
dy = lorenz(y, sigma, beta, rho)
@assert length(dy) == length(y)

y = randn(3,10000)
y = (y[1,:], y[2,:], y[3,:])
@time dy = lorenz(y, sigma, beta, rho)

# Speed Tests
# Single function call - 0.000006 seconds
# Vectorized function call method 1 - 0.000404 seconds
# Vectorized function call method 2 - 0.001036 seconds
# Vectorized function call method 3 - 0.000187 seconds
# Note: Why is this 7 times slower than Python?
# (Using: @time dy = lorenz(y, sigma, beta, rho);)
