function P = numeric_leg_fun_vec(N, x)
x_mapped = 2 * x - 1; 
    P = zeros(length(x), N);
    P(:, 1) = 1;  
    P(:, 2) = x_mapped; 
    for n = 2:N-1
        P(:, n+1) = ((2*n - 1) .* x_mapped .* P(:, n) - (n - 1) * P(:, n-1)) / n;
    end
    for n = 0:N-1                               
        P(:, n+1) = sqrt(2*n + 1) * P(:, n+1);
    end
end
