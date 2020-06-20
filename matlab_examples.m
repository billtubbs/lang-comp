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

% Function
% Functions must go at end of 
% file or in separate file
function y = power(x, a)
    y = x^a;
end



