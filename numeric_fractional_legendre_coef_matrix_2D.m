function C = numeric_fractional_legendre_coef_matrix_2D(I, n, alpha_x, alpha_y)
% Calculates 2D fractional Legendre coefficients for an image
% Inputs:
%   I - input image (matrix)
%   n - number of basis functions in each direction (total n×n coefficients)
%   alpha_x, alpha_y - fractional orders for x and y directions
% Output:
%   C - n×n coefficient matrix

if nargin < 4
    alpha_y = alpha_x; % Use same alpha for both directions if only one given
end

[M, N] = size(I);

% Generate sample points
x_val = linspace(0, 1, M)';
y_val = linspace(0, 1, N)';

% Get 1D fractional Legendre basis functions
Bx = numeric_fractional_leg_fun_vec(n, x_val, alpha_x);
By = numeric_fractional_leg_fun_vec(n, y_val, alpha_y);

% Initialize coefficient matrix
C = zeros(n, n);

% Integration steps
dx = 1/(M-1);
dy = 1/(N-1);

% Weight functions
wx = x_val.^(alpha_x - 1);
wy = y_val.^(alpha_y - 1);

% Calculate coefficients
for i = 0:n-1
    for j = 0:n-1
        % 2D basis function (separable)
        Phi_ij = Bx(:, i+1) * By(:, j+1)';
        
        % Weight matrix
        W = wx * wy';
        
        % Integrand
        integrand = I .* Phi_ij .* W;
        
        % Numerical integration
        if alpha_x < 1 || alpha_y < 1
            % Handle singularity at boundaries
            % Use midpoint rule for boundary strips
            C(i+1, j+1) = sum(sum(integrand(2:end, 2:end))) * dx * dy;
        else
            % Standard 2D trapezoidal integration
            C(i+1, j+1) = trapz(y_val, trapz(x_val, integrand, 1), 2);
        end
    end
end

end