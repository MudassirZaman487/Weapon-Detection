function F = numeric_fractional_leg_fun_vec(N, x, alpha)
    % Computes FLFs using the explicit series formula:
    %   FL_i^alpha(x) = sum_{s=0}^{i} b_{s,i} * x^(s * alpha)
    % Inputs:
    %   N     - number of FLFs to compute
    %   x     - column vector of x values in [0,1]
    %   alpha - fractional parameter (alpha > 0)
    % Output:
    %   F     - matrix of size (length(x), N), each column is FL_i^alpha(x)

    x = x(:);                      % Ensure x is column vector
    F = zeros(length(x), N);      % Preallocate output

    for i = 0:N-1
        FL_i = zeros(size(x));    % FL_i^alpha(x) for i-th basis function
        for s = 0:i
            coeff = ((-1)^(i + s) * factorial(i + s)) / ...
                    (factorial(i - s) * (factorial(s))^2);
            FL_i = FL_i + coeff * x.^(s * alpha);
        end
        F(:, i+1) = alpha*sqrt(2*i + 1) * FL_i;  % Normalization
    end
end