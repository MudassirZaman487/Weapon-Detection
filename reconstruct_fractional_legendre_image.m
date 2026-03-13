function I_recon = reconstruct_fractional_legendre_image(C, n, x_vals, y_vals, alpha_x, alpha_y)
% Reconstructs image from fractional Legendre coefficients
% Inputs:
%   C - n×n coefficient matrix
%   n - number of basis functions in each direction
%   x_vals, y_vals - sample points for reconstruction
%   alpha_x, alpha_y - fractional orders for x and y directions
% Output:
%   I_recon - reconstructed image

if nargin < 6
    alpha_y = alpha_x; % Use same alpha for both directions if only one given
end

M = length(x_vals);
N = length(y_vals);

% Get 1D fractional Legendre basis functions
Bx = numeric_fractional_leg_fun_vec(n, x_vals, alpha_x);
By = numeric_fractional_leg_fun_vec(n, y_vals, alpha_y);

% Initialize reconstructed image
I_recon = zeros(M, N);

% Reconstruct by summing basis functions weighted by coefficients
for i = 0:n-1
    for j = 0:n-1
        % Add contribution from each basis function
        I_recon = I_recon + C(i+1, j+1) * (Bx(:, i+1) * By(:, j+1)');
    end
end

end