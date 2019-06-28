function [logl, KL, auc, logls, KLs, aucs, time] = prlp(train,test, ts_mean, ts_var, cfg)

threshold = cfg.tol;
rho = cfg.rho;
max_iter = cfg.max_iter;

y = train.y;
x = train.x;

N = size(y, 1);
M = size(test.y, 1);
D = size(x, 2);
iter = 1;

% record
logls = [];
KLs = [];
aucs = [];
time = [];

% intermediate record
intermediate_record = struct('joint_var', {}, 'joint_mean', {}, 'time', {});

% initialize
% mean_list = zeros(N, D);
precision_list = ones(N,D) * 1e-6;
prod_list = zeros(N, D);
new_precision = zeros(N,D);
new_prod = zeros(N, D);
new_mean = zeros(N,D);


% calculate joint
joint_precision = sum(precision_list) + 1;
joint_prod = sum(prod_list);
joint_mean = (1 ./ joint_precision) .* joint_prod;

tic;
while true
    q_precision = repmat(joint_precision, N, 1) - precision_list;
    tmp = repmat(joint_precision .* joint_mean, N, 1);
    q_var = 1 ./ q_precision;
    q_mean = q_var .* (tmp - prod_list);

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
%          options.DERIVATIVECHECK = 'on';
        func = @(theta) lpfun(theta, param);
        [opt_theta] = minFunc(func, theta0, options);
        
        % second order
        tx = (2 * y(i) - 1) * x(i, :) * opt_theta;
        pdc = exp(mvnormpdfln(tx) - normcdfln(tx));
%         pdc = normpdf(tx) / normcdf(tx);
        tmp = x(i, :) * opt_theta * pdc + (2 * y(i) - 1) * pdc^2;
        
        new_precision(i, :) = (2* y(i) - 1) * tmp * (x(i, :).^2);
        
        tmp = find(new_precision(i, :) <= 0);
        new_precision(i,tmp) = 1e-6;
        
        new_joint_mean = opt_theta';
        new_joint_precision =  new_precision(i, :) + q_precision(i,:);
%         new_precision(i, :) = new_joint_precision - q_precision(i, :);
        new_prod(i, :) = new_joint_precision .* new_joint_mean - q_precision(i, :) .* q_mean(i, :);
%         new_mean(i, :) = 1 ./ new_precision(i,:) .* (new_joint_precision .* new_joint_mean - q_precision(i, :) .* q_mean(i, :));
    end
%     % correction
%         tmp = find(new_precision <=0 | isnan(new_precision) | isinf(new_precision));
%         new_precision(tmp) = 1e-6;
%         new_mean(tmp) = 0; %mean_list(tmp);
    if iter == 390
        pause;
    end

    time(iter) = toc;
    
    precision_list = precision_list * (1 - rho) + new_precision * rho;
    old_prod = prod_list;
    prod_list = prod_list * (1 - rho) + new_prod * rho;
%     old_mean = mean_list;
%     mean_list = mean_list * (1 - rho) + new_mean * rho;

    diff = mean(sum((old_prod - prod_list).^2, 2));

    % calculate joint
    joint_precision = sum(precision_list) + 1;
    joint_prod = sum(prod_list);
    joint_mean = (1 ./ joint_precision) .* joint_prod;
    
    % KL
    joint_var = diag(1 ./ joint_precision);    
    KL = 0.5 * (logdet(joint_var) -  logdet(ts_var) + trace(joint_var \ ts_var) + (joint_mean' - ts_mean)' * inv(joint_var) * (joint_mean' - ts_mean) - D);
    
    % log likelihood
    
    t1 = (2* y - 1) .* (x * joint_mean');
    t2 = sqrt(1 + x.^2 * (1./ joint_precision)');
    train_logl = mean(normcdfln(t1 ./ t2));

    t1 = (2* test.y - 1) .* (test.x * joint_mean');
    t2 = sqrt(1 + test.x.^2 * (1./ joint_precision)');
    
    p = normcdf(test.x * joint_mean');
    [~,~,~,auc] = perfcurve(test.y,p,1);
    
    tmp = normcdfln(t1 ./ t2);
    logl = mean(tmp);
    
    % record
    KLs(end+1) = KL;
    logls(end+1) = logl;
    aucs(end+1) = auc;   
    
    r.joint_var = joint_var;
    r.joint_mean = joint_mean;
    r.time = time(iter);
    intermediate_record(end+1) = r;
    
    disp(sprintf('prlp -- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl ,auc, KL));
    if diff < threshold || iter > max_iter 
        save('./intermediate/prlp.mat', 'intermediate_record');
        break;
    end
    iter = iter + 1;
end

end

