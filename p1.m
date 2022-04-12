%% (b) and (c)
clc; clear;

load P1.mat

M = size(A,1);
N = size(A,2);

Method_str = ["Gradient", "Newton"];

Max_len = 1e5;
F_x = nan(1,Max_len); % Record f(x*)
T = nan(1,Max_len); % Record t
alg_mode = 1; % 1 is gradient method. 2 is newton method
bound = [1e-3, 1e-8]; % Bound that is for judge if the iteration have to stop.

f = @(x) -sum(log(1-A*x),1) - sum(log(1-x.^2),1);
grad_f = @(x) transpose(sum(diag(1./(1-A*x))*A, 1)) - (1./(1+x) - 1./(1-x));

alp = 0.2;
bta =  0.1;
x = zeros(N,1); % initial x0

if (alg_mode == 1)
    indicator = norm(grad_f(x))
elseif (alg_mode == 2)
    indicator = 1; % Get Lambda^2
end

k=1;

while (indicator> bound(alg_mode))
    t=1;
    gfx = grad_f(x);
    fx = f(x);
    
    %Choose step size according to the method.
    if (alg_mode == 1)
        d_x = -gfx;
    elseif (alg_mode == 2)
        d_x = -hes_f_p1(A,x)\gfx;
    end
    norm_now = norm(d_x) % debug
    
    %Backtracking
    while (any([1-A*(x+t*d_x);1-(x+t*d_x).^2] < 0)) % Search the t that make x+ be in the domain of f.
        t = bta*t;
    end
    
    temp = transpose(grad_f(x))*d_x; % Prevent unecessary calculations.
    while ( f(x+t*d_x) > fx+alp*t*temp)
        t = bta*t;
    end
    x = x+t*d_x;
    
    if (alg_mode == 1)
        indicator = norm(grad_f(x));
    elseif (alg_mode == 2)
        indicator = transpose(grad_f(x))*(hes_f_p1(A,x)\grad_f(x)); % Get Lambda^2
    end
    
    if (k <= Max_len)
        F_x(k) = f(x);
        T(k) = t;
    else
        F_x(end+1) = f(x);
        T(end+1) = t;
    end
    k = k+1;
end

p_optim = f(x)

figure(1)
F_x(isnan(F_x)) = [];
plot(F_x - p_optim);
title(Method_str(alg_mode) + ": f(x^{(k)})-p* in each iteration. \alpha ="+num2str(alp)+" \beta ="+num2str(bta));
xlabel("Iteration");
ylabel("f(x^{(k)})-p*")

figure(2)
T(isnan(T)) = [];
plot(T,'-x');
title(Method_str(alg_mode) + ": t^{(k)} in each iteration. \alpha ="+num2str(alp)+" \beta ="+num2str(bta));
xlabel("Iteration");
ylabel("Step size t^{(k)}");