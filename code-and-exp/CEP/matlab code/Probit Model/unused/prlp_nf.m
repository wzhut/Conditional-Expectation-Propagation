function [logl, KL, auc, logls, KLs, aucs, time] = prlp_nf(train,test, ts_mean, ts_var, cfg)

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
mean_list = rand(D, 1, N);
precision_list = repmat(eye(D) * 1e-6, [1,1,N]);
prod_list = for_multiprod(precision_list, mean_list);
new_precision = zeros(D,D,N);
new_prod = zeros(D,1,N);


% calculate joint
joint_precision = sum(precision_list, 3) + eye(D);
joint_var = inv(joint_precision);
joint_prod = sum(prod_list,3);
joint_mean = joint_var * joint_prod;

tic;
while true
    % calculate q
    q_precision = repmat(joint_precision,[1,1,N]) - precision_list;
    q_prod = repmat(joint_prod,[1,1,N]) - prod_list;
    q_var = for_multinv(q_precision);
    q_mean = for_multiprod(q_var, q_prod);

    % replace each likelihood
    for i = 1 : N
        param.x = x(i,:);
        param.y = y(i);
        param.mu = q_mean(:,:,i);
        param.precision = q_precision(:,:,i);
        theta0 = zeros(D,1);
        options = [];
        options.display = 'none';
        options.Method = 'lbfgs';
%         options.DERIVATIVECHECK = 'on';
        func = @(theta) lpfun_nf(theta, param);
        [opt_theta] = minFunc(func, theta0, options);
        
        % second order
        tx = (2 * y(i) - 1) * x(i, :) * opt_theta;
        
        pdc = normpdf(tx) / normcdf(tx);
        tmp = x(i, :) * opt_theta * pdc + (2 * y(i) - 1) * pdc^2;
        new_precision(:, :, i) = (2* y(i) - 1) * tmp * x(i,:)' * x(i,:);
        [~, p] = chol(new_precision(:, :, i));
        if rank(new_precision(:, :, i)) < D || p > 0
            new_precision(:, :, i) = eye(D) * 1e-6;
        end
        new_joint_mean = opt_theta;
        new_joint_precision =  new_precision(:, :, i) + q_precision(:, :, i);
        new_prod(:, :, i) = new_joint_precision * new_joint_mean - q_precision(:, :, i) * q_mean(:, :, i);
    end
    
    time(iter) = toc;
    
    precision_list = precision_list * (1 - rho) + new_precision * rho;
    old_prod = prod_list;
    prod_list = prod_list * (1 - rho) + new_prod * rho;
    % calculate joint
    joint_precision = sum(precision_list, 3) + eye(D);
    joint_var = inv(joint_precision);
    joint_prod = sum(prod_list,3);
    joint_mean = joint_precision \ joint_prod;
    
    diff = sum(sum((squeeze(prod_list - old_prod)).^2, 2));
    
    % KL  
    KL = 0.5 * (logdet(joint_var) -  logdet(ts_var) + trace(joint_var \ ts_var) + (joint_mean - ts_mean)' / joint_var * (joint_mean - ts_mean) - D);
    
    % log likelihood
    
    t1 = (2* y - 1) .* (x * joint_mean);
    tmp = zeros(N,1);
    t2 = sqrt(1 + diag(x / joint_precision * x'));
    train_logl = mean(normcdfln(t1 ./ t2));

    t1 = (2* test.y - 1) .* (test.x * joint_mean);
    t2 = sqrt(1 + diag(test.x / joint_precision * test.x'));
    
    p = normcdf(test.x * joint_mean);
    [~,~,~,auc] = perfcurve(test.y,p,1);
    
    tmp = normcdfln(t1 ./ t2);
    logl.mean = mean(tmp);
    logl.std = std(tmp);
    
    % record
    KLs(end+1) = KL;
    logls(end+1) = logl;
    aucs(end+1) = auc;
    
    r.joint_var = joint_var;
    r.joint_mean = joint_mean;
    r.time = time(iter);
    intermediate_record(end+1) = r;
    
    disp(sprintf('prlp_nf -- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl.mean ,auc, KL));
    if diff < threshold || iter > max_iter 
        save('./intermediate/prlp_nf.mat', 'intermediate_record');
        break;
    end
    iter = iter + 1;
end

end

