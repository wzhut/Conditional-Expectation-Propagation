function [logl, KL, auc, logls, KLs, aucs, time] = lrep_nf(train,test, ts_mean, ts_var, cfg)

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
    % calculate q
    q_precision = repmat(joint_precision,[1,1,N]) - precision_list;
    q_prod = repmat(joint_prod,[1,1,N]) - prod_list;
    q_var = for_multinv(q_precision);
    q_mean = for_multiprod(q_var, q_prod);
    
    v_i = zeros(N, 1);
    m_i = zeros(N, 1);
    for i = 1 : N
        v_i(i) = x(i, :) * q_var(:,:,i) * x(i,:)';
        m_i(i) = x(i, :) * q_mean(:,:,i);
    end
    
    eta = train_nd .* repmat(sqrt(v_i), [1, num_nd]) + repmat(m_i, [1, num_nd]);
    sigma_tx = sigma(eta);
    
    h = sigma_tx;
    idx0 = find(train_yn == 0);
    h(idx0) = 1 - h(idx0);
    
    E0 = h * weight;
    E1 = (h .* eta) * weight;
    E2 = (h .* eta.^2) * weight;

    m = E1 ./ E0;
    v = E2 ./ E0 - m.^2;
    
    mdv =  m ./ v - m_i ./ v_i;
    rv = 1 ./ v - 1./v_i;
    
    for i = 1 : N
        new_precision(:,:,i) = rv(i) * x(i,:)' * x(i,:);
        new_prod(:,:,i) = mdv(i) * x(i,:)';
%         if rank(new_precision_list(:,:,m,i)) < r
%             new_precision_list(:,:,m,i) = eye(r) * 1e-6;
%         end
    end
    
    time(iter) = toc;
    
    precision_list = precision_list * (1 - rho) + new_precision * rho;
    old_prod = prod_list;
    prod_list = prod_list * (1 - rho) + new_prod * rho;
    % calculate joint
    joint_precision = sum(precision_list, 3) + eye(D);
    joint_var = inv(joint_precision);
    joint_prod = sum(prod_list,3);
    joint_mean = joint_var * joint_prod;
    
    diff = sum(sum((squeeze(prod_list - old_prod)).^2, 2));
    
    % KL   
    KL = 0.5 * (logdet(joint_var) -  logdet(ts_var) + trace(joint_var \ ts_var) + (joint_mean - ts_mean)' * inv(joint_var) * (joint_mean - ts_mean) - D);
    
    % log-likelihood
    % train
    proj_mean = x * joint_mean;
    proj_var = diag(x / joint_var * x');
    theta_s = train_nd .* (repmat(sqrt(proj_var), [1, num_nd])) + repmat(proj_mean, [1, num_nd]);
    sigma_tx = sigma(theta_s);
    h = sigma_tx;
    idx0 = find(train_yn == 0);
    h(idx0) = 1 - h(idx0);
    train_logl = mean(log(h * weight + realmin));
    % test
    proj_mean = test.x * joint_mean;
    proj_var = diag(test.x / joint_var * test.x');
    theta_s = test_nd .* (repmat(sqrt(proj_var), [1, num_nd])) + repmat(proj_mean, [1, num_nd]);
    sigma_tx = sigma(theta_s);
    h = sigma_tx;
    idx0 = find(test_yn == 0);
    h(idx0) = 1 - h(idx0);
    tmp = log(h * weight + realmin);
    logl.mean = mean(tmp);
    logl.std = std(tmp);
    
    p = sigma(test.x * joint_mean);
    [~,~,~,auc] = perfcurve(test.y,p,1);
    
    % record
    KLs(end+1) = KL;
    logls(end+1) = logl;
    aucs(end+1) = auc;
    
    r.joint_var = joint_var;
    r.joint_mean = joint_mean;
    r.time = time(iter);
    intermediate_record(end+1) = r;
    
    disp(sprintf('lrep_nf -- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl.mean ,auc, KL));
    if diff < threshold || iter > max_iter 
        save('./intermediate/lrep_nf.mat', 'intermediate_record');
        break;
    end
    iter = iter + 1;
end

end

