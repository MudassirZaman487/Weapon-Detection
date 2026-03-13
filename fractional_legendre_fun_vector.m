function P = fractional_legendre_fun_vector(x, M, alpha)
% Generates symbolic fractional-order Legendre functions
% FL_i^alpha(x) = sum_{s=0}^{i} b_{s,i} * x^{s*alpha}
% Inputs:
%   x - symbolic variable
%   M - number of polynomials (0 to M-1)
%   alpha - fractional order parameter
% Output:
%   FL - column vector of M fractional Legendre polynomials

P = sym(zeros(M, 1));
syms k 
    P = sym([]);
    for i = 0:M-1
        aa = ((-1)^(k+i)* gamma(k+i+1));
        bb = (gamma(i-k+1)*(gamma(k+1))*gamma(k+1));
        P(i+1,1) = symsum((aa/bb)*x^(k*alpha),k,0,i);
    end

    for n = 0:M-1

    P(n+1) = sqrt((2*n+1)*alpha)*P(n+1);
    end
end
