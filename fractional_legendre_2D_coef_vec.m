function C = fractional_legendre_2D_coef_vec(f, M, x, y, alpha_x, alpha_y)
% Calculates coefficients for 2D fractional Legendre expansion
% Inputs:
%   f - symbolic function f(x,y) to approximate
%   M - order (will generate M×M coefficients)
%   x, y - symbolic variables
%   alpha_x, alpha_y - fractional orders for x and y directions
% Output:
%   C - row vector of M^2 coefficients

if nargin < 6
    alpha_y = alpha_x; % Use same alpha for both directions if only one given
end

% Get 2D basis functions
product = fractional_legendre_2D_fun_vec(M, x, y, alpha_x, alpha_y);

% Weight functions
w_x = x^(alpha_x - 1);
w_y = y^(alpha_y - 1);
w_xy = w_x * w_y;

% Calculate integral with weight function
integral = int(int(f * product * w_xy, x, 0, 1), y, 0, 1);

% Calculate coefficients
C = sym(zeros(1, M^2));
for a = 0:M-1
    for b = 0:M-1
        count = M*a + b + 1;
        % Normalization factor
        norm_factor = alpha_x * alpha_y * (2*a + 1) * (2*b + 1);
        C(count) = norm_factor * integral(count);
    end
end

end