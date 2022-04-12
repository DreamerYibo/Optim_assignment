%%Caculate hessian matrix in p1.
function  Hes = hes_f_p1(A, x)

M = size(A,1);
N = size(A,2);
hes_temp = zeros(N,N);
for k=1:M
    hes_temp = hes_temp + (transpose(A(k,:))*A(k,:))/(1-A(k,:)*x)^2;%first term
end
Hes=hes_temp+diag(1./((1+x).^2) + 1./((1-x).^2));
end
