function [logl, KL, auc, logls, KLs, aucs, time] = lrepis(train, test, ts_mean, ts_var)

threshold = 1e-5;
rho = 0.05;
y = train.y;
x = train.x;
N = size(y,1);
M = size(test.y, 1);
D = size(x,2);
iter = 1;
max_iter = 10000;

logls = struct('mean', {}, 'std', {});
KLs = [];
aucs = [];
time = [];


% initialize
mean_list = zeros(N, D);
precision_list = ones(N,D) * 1e-6;

% calculate joint
joint_precision = sum(precision_list) + 1;
joint_mean = 1 ./ joint_precision .* sum(precision_list .* mean_list);

% std mvn 100000 samples
num_samples = 100000;
g_samples = mvnrnd(zeros(D,1), eye(D), num_samples)';

% gaussian-hermite quadrature
num_nd = 9;
[nd,weight] = quadrl(num_nd);
train_nd = repmat(nd, [N, 1]);
train_yn = repmat(y, [1, num_nd]);

test_nd = repmat(nd, [M, 1]);
test_yn = repmat(test.y, [1, num_nd]);

% sigma function
sigma = @(x) 1 ./ (1 + exp(-x));

% start timer
tic;
while true
    % q
    q_precision = repmat(joint_precision, N, 1) - precision_list;
    tmp = repmat(joint_precision .* joint_mean, N, 1);
    q_var = 1 ./ q_precision;
    q_mean = q_var .* (tmp - precision_list .* mean_list);
    
    new_precision = zeros(N,D);
    new_mean = zeros(N,D);
    
%     f = normcdf((repmat(2 * y - 1, [1, D]) .* x) * g_samples);
    f = sigma((repmat(2 * y - 1, [1, D]) .* x) * g_samples);
    g = zeros(num_samples, N);
    for i = 1 : N
        mu = repmat(q_mean(i,:)', [1, num_samples]);
        v = repmat(q_var(i, :)', [1, num_samples]);
        f(i, :) = f(i,:) .* prod(normpdf(g_samples, mu, v));
    end
    g = repmat(prod(normpdf(g_samples)), [N , 1]);
    w = f ./ g;
    E0 = repmat(mean(w, 2), [1, D]);
    E1 = w * g_samples' / num_samples;
    E2 = w * g_samples'.^2 / num_samples;
    
    new_joint_mean = E1 ./ E0;
    new_joint_var = E2 ./ E0 - new_joint_mean.^2;
    new_joint_precision = 1 ./ new_joint_var;
    
    new_precision = new_joint_precision - q_precision;
    new_mean = 1./ new_precision .* (new_joint_precision .* new_joint_mean - q_precision .* q_mean);
    
    tmp = find(new_precision <=0 | isnan(new_precision) | isinf(new_precision));
    new_precision(tmp) = 1e-6;
    new_mean(tmp) = 0;
    
    
    time(iter) = toc;
    
    precision_list = precision_list * (1 - rho) + new_precision * rho;
    old_mean = mean_list;
    mean_list = mean_list * (1 - rho) + new_mean * rho;
    
    diff = mean(sum((old_mean - mean_list).^2, 2));
    
    % calculate joint
    joint_precision = sum(precision_list) + 1;
    joint_mean = 1 ./ joint_precision .* sum(precision_list .* mean_list);
    
    % KL divergence
    joint_var = diag(1 ./ joint_precision);
    KL = 0.5 * (logdet(joint_var) -  logdet(ts_var) + trace(joint_var \ ts_var) + (joint_mean' - ts_mean)' * inv(joint_var) * (joint_mean' - ts_mean) - D);
    
    % log likelihood
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
    
    p = sigma(sum(test.x .* repmat(joint_mean, [M, 1]), 2));
    [~,~,~,auc] = perfcurve(test.y,p,1);
    
    % record
    KLs(end+1) = KL;
    logls(end+1) = logl;
    aucs(end+1) = auc;
    
    disp(sprintf('lrepis-- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl.mean ,auc, KL)); 
    iter = iter + 1;
    if diff < threshold || iter > max_iter %|| abs(last_train_logl - train_logl) < threshold * 1e-3
        break;
    end
%     last_diff = diff;
%     last_train_logl = train_logl;
end
end

