function C = legendre_coef_matrix1_2D(I, n)

    [M, N] = size(I);
    x_val = linspace(0,1,M);
    y_val = linspace(0,1,N);
    Bx = numeric_leg_fun_vec(n, x_val');
    By = numeric_leg_fun_vec(n, y_val');
    C = zeros(n, n);
    dx = 1/(M-1);
    dy = 1/(N-1);
    for i = 0:n-1
        for j = 0:n-1
            Phi_ij = Bx(:,i+1) * By(:,j+1)';
            integrand = I .* Phi_ij;
            C(i+1, j+1) = sum(sum(integrand)) * dx*dy;        
        end
    end
end
