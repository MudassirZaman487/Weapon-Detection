function FL = fractional_legendre_recur(x, N, alpha)
% Computes symbolic fractional-order Legendre functions using recurrence
% Inputs:
%   x     - symbolic variable
%   N     - number of functions (from FL_0 to FL_{N-1})
%   alpha - fractional order (> 0)
% Output:
%   FL    - column vector of symbolic functions

FL = sym(zeros(N, 1));  % Preallocate symbolic vector

% Base functions
x_alpha = x^alpha;
FL(1) = sym(1);                        % FL_0^(alpha)(x)
FL(2) = 2*x_alpha - 1;             % FL_1^(alpha)(x)

% Recurrence
for k = 1:N-2
    a = (2*k + 1)/(k + 1);
    b = k / (k + 1);
    FL(k+2) = a*(2*x_alpha - 1)*FL(k+1) - b*FL(k);
end

% Apply normalization (orthonormal basis)
for n = 0:N-1
    FL(n+1) = simplify(sqrt((2*n + 1)*alpha) * FL(n+1));
end
end
