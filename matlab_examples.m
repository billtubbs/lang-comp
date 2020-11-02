%% Comparison of high-level programming languages
% This is MATLAB

% Assignment
x = power(2, 0.5);

% Standard output
disp(x)

% Or omit line terminator
x

% Formatted output
fprintf("%5.2f\n", x)

% Definite loop
for i=1:5
  disp(i)
end

% Indefinite loop
i = 1;
while i <= 5
  disp(i)
  i = i + 1;
end

% Dictionary (container.Map in MATLAB)
symbols = {'H','He','Li'};
values = [1 2 3];
elements = containers.Map(symbols,values)

% Iterating over collections
for symbol = keys(elements)
    n = elements(symbol{1});
    fprintf("%s: %d\n", symbol{1}, n)
end

% Array literals
% Vector (1d)
x = [1, 2, 3]

% Matrix (2d)
A = [1 2 3; 4 5 6]

% Nd array (2x3x2)
Z = ones(2, 3, 2)

% Array dimensions
size(x)
size(A)

% Indexing arrays
x(2)
A(2,2)
A(2)

%Slicing arrays
x(2:end)
A(:,2)

% Array broadcasting
1 - A
A + x

% Linear algebra
A = [0.8 0; 0 1]; 
K = [0.942; 1.074]; 
C = [0.2 1];
(eye(2) - A+K*C)^-1

% Symbolic math
syms x y
y = x^2 - 2*x + 1
dydx = diff(y,x)
x_min = solve(dydx,x)

% Vectorizable function
% Parameter values
beta = 8 / 3;
sigma = 10;
rho = 28;

% Test 1 - Single function call
y = [-8; 8; 27];
dy = lorenz(y, sigma, beta, rho);
assert(all(dy == [160; -16; -136]))

% Test 2 - Single function calls within for loop
y = repmat([-8, 8, 27], 10, 1)';
dy = lorenz_loop(y, sigma, beta, rho);
assert(all(all(dy == [160; -16; -136])))

% Test 3 - vectorized function call
dy = lorenz_vec(y, sigma, beta, rho);
assert(all(all(dy == [160; -16; -136])))

disp("Speed tests")
y = [-8; 8; 27];
f = @() lorenz(y, sigma, beta, rho);
t = timeit(f);
fprintf("Test 1: %f seconds\n", t);
n = 10000;
y = randn(3, n);
f = @() lorenz_loop(y, sigma, beta, rho);
t = timeit(f);
fprintf("Test 2: %f seconds\n", t);
f = @() lorenz_vec(y, sigma, beta, rho);
t = timeit(f);
fprintf("Test 3: %f seconds\n", t);
% Speed test results
% Test 1: 0.000000 seconds
% Test 2: 0.002757 seconds 
% Test 3: 0.000096 seconds

% Note: Functions must go at end of file
% or in separate file
function y = power(x, a)
  y = x^a;
end

% Scalar function
function dy = lorenz(y, sigma, beta, rho)
  dy = [sigma * (y(2) - y(1)); ...
        y(1) * (rho - y(3)) - y(2); ...
        y(1) * y(2) - beta * y(3)];
end

% Looped calls
function dy = lorenz_loop(y, sigma, beta, rho)
  dy = zeros(size(y));
  for i = 1:size(y,2)
    dy(:,i) = lorenz(y(:,i), sigma, beta, rho);
  end
end

% Vectorizable function
function dy = lorenz_vec(y, sigma, beta, rho)
  dy = [sigma .* (y(2,:) - y(1,:)); ...
        y(1,:) .* (rho - y(3,:)) - y(2,:); ...
        y(1,:) .* y(2,:) - beta .* y(3,:)];
end
