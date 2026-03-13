function product = fractional_legendre_2D_fun_vec(M, x, y, alpha_x, alpha_y)
% Generates 2D fractional-order Legendre functions as tensor products
% Inputs:
%   M - order (will generate M×M functions)
%   x, y - symbolic variables
%   alpha_x, alpha_y - fractional orders for x and y directions
% Output:
%   product - column vector of M^2 2D basis functions

if nargin < 4
    alpha_x = sym('alpha_x');
    alpha_y = sym('alpha_y');
elseif nargin < 5
    alpha_y = alpha_x; % Use same alpha for both directions if only one given
end

% Get 1D fractional Legendre functions
FL_x = fractional_legendre_fun_vector(x, M, alpha_x);
FL_y = fractional_legendre_fun_vector(y, M, alpha_y);

% Create tensor product
product = sym([]);
for a = 0:M-1
    for b = 0:M-1
        count = M*a + b + 1;
        product(count, 1) = FL_x(a+1) * FL_y(b+1);
    end
end

end