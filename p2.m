%Problem 2
clc; clear;
load P2.mat

Max_len = 1e3;
N = size(W,1);
%v is 20x1 vec. t is scalar.
f = @(v,t) t*sum(v)+(-log(det(W+diag(v))));
grad_f = @(v,t) t*ones(N,1)-diag(inv(W+diag(v)));
hes_f = @(v) (inv(W+diag(v))).^2;

sig = min(eig(W));
alp = 0.01;
bta =  0.5;
v = (1-sig)*ones(N,1); % initial v such that it is feasible.
X_dual = nan(N,N); % dual solution of p2.
t_barrier = 1; % initial t for log barrier fcn.
bound = [1e-6, 1e-8]; %first is for N/t.
indicator = 1;
Fv = nan(1,Max_len); %record f(v,t) in each barrier iteration.
T_barrier = nan(1,Max_len); % record t_barrier.

% %% Exaustive search % Take some time so I comment this and save the values in "P2.mat"
% [p_optim, x_optim] = exaustive_fcn(W)
disp("Exaustive search: x_optim:")
disp(transpose(x_optim))
disp("Exaustive search: optim value:")
disp(p_optim)

k=1;
while (N/t_barrier > bound(1))
    t_barrier = t_barrier*5; % try 5. Need to put this here
    while (indicator > bound(2))
        t = 1; % This t is for newton method.
        gfv = grad_f(v, t_barrier);
        fv = f(v,t_barrier);
        d_v = -hes_f(v)\gfv;
        
        %backtracking
        while ( any(eig(W+diag(v+t*d_v)) < 0)) % Search the t that makes v+ be in the domain of f.
            t = bta*t;
        end
        temp = transpose(grad_f(v,t_barrier))*d_v; % Prevent unecessary calculations.
        while ( f(v+t*d_v, t_barrier) > fv+alp*t*temp)
            t = bta*t;
        end
        
        v = v+t*d_v;
        indicator = transpose(grad_f(v,t_barrier))*(hes_f(v)\grad_f(v,t_barrier)); % Get Lambda^2
    end
    indicator = inf; %reset indicator.

    if (k <= Max_len)
        Fv(k) = f(v,t_barrier);
        T_barrier(k) = t_barrier;
    else
        Fv(end+1) = f(v,t_barrier);
        T_barrier(end+1) = t_barrier;
    end
    k = k+1;
end

d_optim = -sum(v)
X_dual = -1/t_barrier*inv(-(W+diag(v)));
disp("rank of X_dual with default tolerance:")
disp(rank(X_dual))
tol_rank = 1e-4;
disp("rank of X_dual with tolerance " + num2str(tol_rank) + " :")
disp(rank(X_dual,tol_rank))

figure(1)
Fv(isnan(Fv)) = [];
T_barrier(isnan(T_barrier)) = [];
plot((-Fv./T_barrier - d_optim));
title("-f(v^{(k)})/t-d* versus barrier iterations")
xlabel("Iteration")
ylabel("-f(v^{(k)})/t-d*")

%% Simple partition (d)
[V, D] = eig(X_dual);
[~, max_idx] = max(diag(D));
x_hat_d = sign(V(:,max_idx));
p_hat_d = transpose(x_hat_d)*W*x_hat_d
disp("Simple partition: x_hat")
disp(transpose(x_hat_d))
disp("Simple partition: optim value")
disp(p_hat_d)

%% randomized method (e)
R = chol(X_dual);
mu = zeros(N,1);
K = 100; %num of samples
x_hat_e_samples = R'*randn(N,K) + repmat(mu,1,K);
x_hat_e = zeros(N,1);
p_hat_e = inf; % optimal value
for k=1:K
    x_sign = sign(x_hat_e_samples(:,k));
    temp = transpose(x_sign)*W*x_sign;
    if (temp < p_hat_e)
        x_hat_e = x_sign;
        p_hat_e = temp;
    end
end
disp("Randomized method: x_hat")
disp(transpose(x_hat_e))
disp("Randomized method: optim value")
disp(p_hat_e)

%% Greedy refinement (f)
% x0 = ones(N,1);
% x0 = x_hat_d;
% x0 = x_hat_e;
x0 = [1     1    -1    -1     1    -1     1     1    -1     1    -1    -1     1    -1     1     1     1     1     1    -1]';
x_hat_f = x0;


p_hat_f = transpose(x_hat_f)*W*x_hat_f;
reduced_val = -inf;
while(reduced_val<0)
    val_now = transpose(x_hat_f)*W*x_hat_f;
    x_hat_temp = x_hat_f;
    for k=1:N
        x_hat_temp(k) = -x_hat_temp(k);
        val_temp = transpose(x_hat_temp)*W*x_hat_temp;
        if (val_temp < p_hat_f)
            x_hat_f = x_hat_temp;
            p_hat_f = val_temp;
        end
        x_hat_temp(k) = -x_hat_temp(k);%change it back and try next k.
    end
    reduced_val = p_hat_f - val_now;
end

disp("Greedy refinement: x_hat" + " (with x0 = "+ num2str(x0') + ")")
disp(transpose(x_hat_f))
disp("Greedy refinement: optim value" + " (with x0 = "+ num2str(x0') + ")")
disp(p_hat_f)