function I_recon = reconstruct1_legendre_image(C, n, x_vals, y_vals)
    M = length(x_vals);
    N = length(y_vals);
    Bx = numeric_leg_fun_vec(n, x_vals');
    By = numeric_leg_fun_vec(n, y_vals');
    I_recon = zeros(M, N);
    for i = 0:n-1
        for j = 0:n-1
            I_recon = I_recon + C(i+1,j+1) * (Bx(:,i+1) * By(:,j+1)');
        end
    end
end
