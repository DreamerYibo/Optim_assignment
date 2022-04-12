function Grad = grad_f_p1(A,x)
    M = size(A,1);
    N = size(A,2);
    temp = zeros(N,1);
    for k =1:M
        temp = temp+transpose(A(k,:)).*(1/(1-A(k,:)*x));
    end
    Grad = temp - (1./(1+x) - 1./(1-x));
end