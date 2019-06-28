function [logl, KL, auc, logls, KLs, aucs, time] = lrlp_first_iter(train,test, ts_mean, ts_var, cfg)

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
logls = struct('mean', {}, 'std', {});
KLs = [];
aucs = [];
time = [];

% initialize
mean_list = zeros(N, D);
precision_list = ones(N,D) * 1e-6;
new_precision = zeros(N,D);
new_mean = zeros(N,D);


% calculate joint
joint_precision = sum(precision_list) + 1;
joint_mean = (1 ./ joint_precision) .* sum(precision_list .* mean_list);

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
    q_mean = q_var .* (tmp - precision_list .* mean_list);
    old_mean = mean_list;
    % replace each likelihood
    for i = 1 : N
        param.x = x(i,:);
        param.y = y(i);
        param.mu = q_mean(i,:);
        param.v = q_var(i,:);
        theta0 = zeros(D,1);
        options = [];
        options.display = 'none';
        options.Method = 'lbfgs';
%         options.DERIVATIVECHECK = 'on';
        func = @(theta) lpfun(theta, param);
        [opt_theta] = minFunc(func, theta0, options);
        
        % second order
        tx = x(i, :) * opt_theta;
        sigma_tx = sigma(tx);
        new_joint_precision = sigma_tx * (1 - sigma_tx) * x(i, :).^2 + q_precision(i, :);
        new_joint_mean = opt_theta';
        new_precision(i, :) = new_joint_precision - q_precision(i, :);
        new_mean(i, :) = 1 ./ new_precision(i,:) .* (new_joint_precision .* new_joint_mean - q_precision(i, :) .* q_mean(i, :));
        
        tmp = find(new_precision(i, :) <=0 | isnan(new_precision(i, :)) | isinf(new_precision(i, :)));
        new_precision(i, tmp) = 1e-6;
        new_mean(i, tmp) = mean_list(i, tmp);
        
        precision_list(i,:) = precision_list(i, :) * (1 - rho) + new_precision(i, :) * rho;
        mean_list(i, :) = mean_list(i, :) * (1 - rho) + new_mean(i,:) * rho;
        
        if mod(i, 100) == 0
            time(end+1) = toc;
            % calculate joint
            joint_precision = sum(precision_list) + 1;
            joint_mean = 1 ./ joint_precision .* sum(precision_list .* mean_list);

            % KL
            joint_var = diag(1 ./ joint_precision);    
            KL = 0.5 * (logdet(joint_var) -  logdet(ts_var) + trace(joint_var \ ts_var) + (joint_mean' - ts_mean)' / joint_var * (joint_mean' - ts_mean) - D);        
            KLs(end+1) = KL;
        end
    end
    

    % correction
%         tmp = find(new_precision <=0 | isnan(new_precision) | isinf(new_precision));
%         new_precision(tmp) = 1e-6;
%         new_mean(tmp) = mean_list(tmp);
    
    time(end+1) = toc;
    
%     precision_list = precision_list * (1 - rho) + new_precision * rho;
    
%     mean_list = mean_list * (1 - rho) + new_mean * rho;

    diff = mean(sum((old_mean - mean_list).^2, 2));

    % calculate joint
    joint_precision = sum(precision_list) + 1;
    joint_mean = 1 ./ joint_precision .* sum(precision_list .* mean_list);
    
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
    
    p = sigma(sum(test.x .* repmat(joint_mean, [M, 1]), 2));
    [~,~,~,auc] = perfcurve(test.y,p,1);
    
    % record
    KLs(end+1) = KL;
    logls(end+1) = logl;
    aucs(end+1) = auc;
    
%     r.joint_var = joint_var;
%     r.joint_mean = joint_mean;
%     r.time = time(iter);
%     intermediate_record(end+1) = r;
    
    disp(sprintf('lrlp -- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl.mean ,auc, KL));
    if diff < threshold || iter > max_iter 
%         save('./intermediate/lrlp.mat', 'intermediate_record');
        break;
    end
    iter = iter + 1;
end

end

