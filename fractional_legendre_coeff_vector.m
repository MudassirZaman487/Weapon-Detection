function C = fractional_legendre_coeff_vector(f, M, x, alpha)
% Calculates coefficients for fractional Legendre expansion
% f(x) ≈ sum_{i=0}^{M-1} c_i * FL_i^alpha(x)
% Inputs:
%   f - symbolic function to approximate
%   M - number of coefficients
%   x - symbolic variable
%   alpha - fractional order parameter
% Output:
%   C - row vector of M coefficients

% Get fractional Legendre functions
FL = fractional_legendre_fun_vector(x, M, alpha);

% Initialize coefficient vector
C = sym(zeros(1, M));

% Weight function w(x) = x^(alpha-1)
w = x^(alpha - 1);

for i = 0:M-1
    % Calculate coefficient using orthogonality
    % c_i = alpha*(2i+1) * int(FL_i(x) * f(x) * w(x), x, 0, 1)
    integrand = FL(i+1) * f * w;
    C(i+1) = alpha * (2*i + 1) * int(integrand, x, 0, 1);
end

end