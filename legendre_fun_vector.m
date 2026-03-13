
function P = legendre_fun_vector(x,M)

    syms k x
    P = sym([]);
    for i = 0:M-1
        aa = ((-1)^(k+i)* gamma(k+i+1));
        bb = (gamma(i-k+1)*(gamma(k+1))*gamma(k+1));
        P(i+1,1) = symsum((aa/bb)*x^k,k,0,i);
    end
end
