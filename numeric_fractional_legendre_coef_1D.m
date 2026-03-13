function C = numeric_fractional_legendre_coef_1D(f_vals, M, alpha)
% Calculates coefficients for fractional Legendre expansion numerically
% f(x) ≈ sum_{i=0}^{M-1} c_i * FL_i^alpha(x)
% Inputs:
%   f_vals - vector of function values at points x
%   M - number of coefficients
%   alpha - fractional order parameter
% Output:
%   C - vector of M coefficients

% Assume f_vals are evaluated at equally spaced points in [0,1]
N = length(f_vals);
x_vals = linspace(0, 1, N)';
dx = 1/(N-1);

% Get fractional Legendre functions at x_vals
FL = numeric_fractional_leg_fun_vec(M, x_vals, alpha);

% Weight function w(x) = x^(alpha-1)
w = x_vals.^(alpha - 1);

% Initialize coefficient vector
C = zeros(M, 1);

% Calculate coefficients using numerical integration
for i = 0:M-1
    % c_i = int(FL_i(x) * f(x) * w(x), x, 0, 1)
    % Note: FL already includes normalization sqrt((2i+1)*alpha)
    integrand = FL(:, i+1) .* f_vals(:) .* w;
    
    % Trapezoidal integration, handling potential singularity at x=0
    if alpha < 1
        % For alpha < 1, w(0) = 0^(alpha-1) = infinity
        % Use midpoint rule for first interval
        C(i+1) = integrand(2) * dx;
        % Trapezoidal for the rest
        C(i+1) = C(i+1) + trapz(x_vals(2:end), integrand(2:end));
    else
        % Standard trapezoidal integration
        C(i+1) = trapz(x_vals, integrand);
    end
end

end