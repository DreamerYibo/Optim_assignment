function [p_optim, x_optim] = exaustive_fcn(W)
%get optim p for p2
N = size(W,1);
p = inf;
p_optim = inf;
x = zeros(N,1);
x_optim = zeros(N,1);

for k=1:2^N
    index = dec2bin(k-1, N);
    for j=1:N 
        if (index(j) == '1')
            x(j) = 1;
        else
            x(j) = -1;
        end
    end
    temp = transpose(x)*W*x;
    if (p > temp)
        p = temp;
        x_optim = x;
    end
end
p_optim = p;
end