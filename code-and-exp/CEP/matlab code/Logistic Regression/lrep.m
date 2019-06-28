function [logl, KL, auc, logls, KLs, aucs, time] = lrep(train,test, ts_mean, ts_var, cfg)

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

tic;
while true
    q_precision = repmat(joint_precision, N, 1) - precision_list;
    tmp = repmat(joint_precision .* joint_mean, N, 1);
    q_var = 1 ./ q_precision;
    q_mean = q_var .* (tmp - prod_list);


    mni = sum(x .* q_mean, 2);
    vni = sum(x.^2 .* q_var, 2);
    
    theta_s = train_nd .* repmat(sqrt(vni), [1, num_nd]) + repmat(mni, [1, num_nd]);
    sigma_tx = sigma(theta_s);
    
    h = sigma_tx;
    idx0 = find(train_yn == 0);
    h(idx0) = 1 - h(idx0);
    
    E0 = h * weight;
    E1 = (h .* theta_s) * weight;
    E2 = (h .* theta_s.^2) * weight;

    m = E1 ./ E0;
    v = E2 ./ E0 - m.^2;
    
    mdv =  m ./ v - mni ./ vni;
    rv = 1 ./ v - 1./vni;
    
    for i = 1 : N
        new_precision(i,:) = rv(i) * diag(x(i,:)' * x(i,:));
        new_prod(i, :) = x(i, :) .* mdv(i);
%         new_mean(i, :) = 1./ new_precision(i, :) .* x(i, :) .* mdv(i);
    end

    % correction
        tmp = find(new_precision <=0 | isnan(new_precision) | isinf(new_precision));
        new_precision(tmp) = 1e-6;
        new_prod(tmp) = prod_list(tmp);
%         new_mean(tmp) = mean_list(tmp);
    
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
    logl.mean = mean(tmp);
    logl.std = std(tmp);
    
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
    
    disp(sprintf('lrep -- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl.mean ,auc, KL));
    if diff < threshold || iter > max_iter
        save('./intermediate/lrep.mat', 'intermediate_record');
        break;
    end
    iter = iter + 1;
end

end

