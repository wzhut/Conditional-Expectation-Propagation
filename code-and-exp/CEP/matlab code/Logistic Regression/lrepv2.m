%use double integration
function [logl, KL, auc, logls, KLs, aucs, time] = lrepv2(train,test, ts_mean, ts_var, cfg)

threshold = cfg.tol;
rho = cfg.rho;
max_iter = cfg.max_iter;

y = train.y;
x = train.x;

N = size(y, 1);
M = size(test.y, 1);
D = size(x, 2);
iter = 1;

% intermediate record
intermediate_record = struct('joint_var', {}, 'joint_mean', {}, 'time', {});

% record
logls = [];
KLs = [];
aucs = [];
time = [];

% initialize
precision_list = ones(N,D) * 1e-6;
prod_list = zeros(N, D);
new_precision = zeros(N,D);
new_prod = zeros(N,D);


% calculate joint
joint_precision = sum(precision_list) + 1;
joint_prod = sum(prod_list);
joint_mean = (1 ./ joint_precision) .* joint_prod;

% gaussian-hermite quadrature
num_nd = 9;
[nd,weight] = quadrl(num_nd);
train_nd = repmat(nd, [N, 1]);
train_yn = repmat(y, [1, num_nd]);

test_nd = repmat(nd, [M, 1]);
test_yn = repmat(test.y, [1, num_nd]);
% sigma function
sigma = @(x) 1 ./ (1 + exp(-x));

[nd, weights] = quadrl(9);
[nd1, nd2] = meshgrid(nd', nd');
nd = [nd1(:), nd2(:)];
[w1, w2] = meshgrid(weights, weights);
weights = w1(:).*w2(:);
    

tic;
while true
    q_precision = repmat(joint_precision, N, 1) - precision_list;
    tmp = repmat(joint_precision .* joint_mean, N, 1);
    q_var = 1 ./ q_precision;
    q_mean = q_var .* (tmp - prod_list);

    %use double integration
    z_each_mean = q_mean.*x;
    z_each_var = q_var.*(x.^2);
    z_each_mean_not = repmat(sum(z_each_mean, 2), [1, D]) - z_each_mean;
    z_each_var_not = repmat(sum(z_each_var,2), [1, D]) - z_each_var;
    new_z_mean = zeros(N,D);
    new_z_var = zeros(N,D);
    for k=1:D
        for n=1:N
            x1 = z_each_mean(n,k) + sqrt(z_each_var(n,k))*nd(:,1);
            x2 = z_each_mean_not(n,k) + sqrt(z_each_var_not(n,k))*nd(:,2);
            E0 = sum(weights./(1 + exp(-(2*y(n)-1)*(x1+x2))));
            E1 = sum(weights.*x1./(1 + exp(-(2*y(n)-1)*(x1+x2))));
            E2 = sum(weights.*(x1.^2)./(1 + exp(-(2*y(n)-1)*(x1+x2))));
            new_z_mean(n,k) = E1/E0;
            new_z_var(n,k) = E2/E0 - (E1/E0)^2;
        end
    end
%     
%     for k=1:D
%         for n=1:N
%             fun0 = @(a,b) normpdf(a,z_each_mean(n,k), sqrt(z_each_var(n,k)))...
%                 .*normpdf(b,z_each_mean_not(n,k), sqrt(z_each_var_not(n,k)))...
%                 ./(1 + exp(-(2*y(n) - 1)*(a+b)));
%             fun1 = @(a, b) a.*fun0(a,b);
%             fun2 = @(a,b) a.^2.*fun0(a,b);
%             E0 = integral2(fun0,  z_each_mean(n,k)-10*sqrt(z_each_var(n,k)),...
%                 z_each_mean(n,k)+10*sqrt(z_each_var(n,k)), ...
%                 z_each_mean_not(n,k) - 10*sqrt(z_each_var_not(n,k)), ...
%                 z_each_mean_not(n,k) + 10*sqrt(z_each_var_not(n,k)));
%             E1 = integral2(fun1,  z_each_mean(n,k)-10*sqrt(z_each_var(n,k)),...
%                 z_each_mean(n,k)+10*sqrt(z_each_var(n,k)), ...
%                 z_each_mean_not(n,k) - 10*sqrt(z_each_var_not(n,k)), ...
%                 z_each_mean_not(n,k) + 10*sqrt(z_each_var_not(n,k)));
%             E2 = integral2(fun2,  z_each_mean(n,k)-10*sqrt(z_each_var(n,k)),...
%                 z_each_mean(n,k)+10*sqrt(z_each_var(n,k)), ...
%                 z_each_mean_not(n,k) - 10*sqrt(z_each_var_not(n,k)), ...
%                 z_each_mean_not(n,k) + 10*sqrt(z_each_var_not(n,k)));
%             new_z_mean(n,k) = E1/E0;
%             new_z_var(n,k) = E2/E0 - (E1/E0)^2;
%         end
%     end
    inv_v = 1 ./ new_z_var - 1./z_each_var;
    inv_v_m  =  new_z_mean ./ new_z_var - z_each_mean ./ z_each_var;
    new_precision = x.^2.*inv_v;
    new_prod = x.*inv_v_m;
%     new_mean = 1./new_precision.*(x.*inv_v_m);
    

    % correction
    tmp = find(new_precision <=0 | isnan(new_precision) | isinf(new_precision));
    new_precision(tmp) = 1e-6;
    new_prod(tmp) = prod_list(tmp);
%     new_mean(tmp) = mean_list(tmp);
    
    time(iter) = toc;
    
    precision_list = precision_list * (1 - rho) + new_precision * rho;
    old_prod = prod_list;
    prod_list = prod_list * (1 - rho) + new_prod * rho;
    diff = mean(sum((old_prod - prod_list).^2, 2));

    % calculate joint
    joint_precision = sum(precision_list) + 1;
    joint_prod = sum(prod_list);
    joint_mean = (1 ./ joint_precision) .* joint_prod;

    % KL
    joint_var = diag(1 ./ joint_precision);    
    KL = 0.5 * (logdet(joint_var) -  logdet(ts_var) + trace(joint_var \ ts_var) + (joint_mean' - ts_mean)' * inv(joint_var) * (joint_mean' - ts_mean) - D);
    
    % log-likelihood
    % train
    proj_mean = x * joint_mean';
    proj_var = (x.^2) * (1 ./ joint_precision)';
    theta_s = train_nd .* (repmat(sqrt(proj_var), [1, num_nd])) + repmat(proj_mean, [1, num_nd]);
    sigma_tx = sigma(theta_s);
    h = sigma_tx;
    idx0 = find(train_yn == 0);
    h(idx0) = 1 - h(idx0);
    train_logl = mean(log(h * weight + realmin));
    % test
    proj_mean = test.x * joint_mean';
    proj_var = (test.x.^2) * (1 ./ joint_precision)';
    theta_s = test_nd .* (repmat(sqrt(proj_var), [1, num_nd])) + repmat(proj_mean, [1, num_nd]);
    sigma_tx = sigma(theta_s);
    h = sigma_tx;
    idx0 = find(test_yn == 0);
    h(idx0) = 1 - h(idx0);
    tmp = log(h * weight + realmin);
    logl = mean(tmp);
    
    p = sigma(test.x * joint_mean');
    [~,~,~,auc] = perfcurve(test.y,p,1);
    
    % record
    KLs(end+1) = KL;
    logls(end+1) = logl;
    aucs(end+1) = auc;
    
    r.joint_var = joint_var;
    r.joint_mean = joint_mean;
    r.time = time(iter);
    intermediate_record(end+1) = r;
    
    disp(sprintf('lrepv2 -- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl ,auc, KL));
    if diff < threshold || iter > max_iter
        save('./intermediate/lrepv2.mat', 'intermediate_record');
        break;
    end
    iter = iter + 1;
end

end

