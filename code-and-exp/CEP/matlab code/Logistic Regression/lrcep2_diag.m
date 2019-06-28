function [logl, KL, auc, logls, KLs, aucs, time] = lrcep2_diag(train,test, ts_mean, ts_var, cfg)

threshold = cfg.tol;
rho = cfg.rho;
max_iter = cfg.max_iter;
cep_iter = cfg.cep_iter;
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
tr = zeros(N,D);

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

    for d = 1 : D
        idx = setdiff(1:D,d);
        tx = sum((repmat(joint_mean(idx), [N, 1]) .* x(:,idx)),2); % + x(:,d) .* q_mean(:,d);
        theta_s = train_nd .* repmat(sqrt(q_var(:,d)), [1, num_nd]) + repmat(q_mean(:,d), [1, num_nd]);
        tx = repmat(tx, [1, num_nd]) + repmat(x(:,d), [1, num_nd]) .* theta_s;
        sigma_tx = sigma(tx);
        
        h = sigma_tx;
        idx0 = find(train_yn == 0);
        h(idx0) = 1 - h(idx0);
        
        E0 = h * weight;
        E1 = (h .* theta_s) * weight;
        E2 = (h .* (theta_s.^2)) * weight;
        
        ts = (1 - 2 * sigma_tx) .* repmat(E0, [1, num_nd]) + 2 * repmat((h .* sigma_tx) * weight, [1, num_nd]);
        
        shared = h .* sigma_tx .* ts;
        t1 = E1 .* (shared  * weight);
        t2 = E0 .* ((shared .* theta_s) * weight);
        t3 = E2 .* (shared * weight);
        t4 = E0 .* ((shared.* (theta_s .^ 2)) * weight);
        
        shared = h .* sigma_tx;
        t5 = E1 .* (shared * weight);
        t6 = E0 .* ((shared .* theta_s) * weight);
        
        D1 = (t5 - t6) ./ (E0.^2);
        
        H1 = (t1 - t2) ./ (E0.^3);
        H2 = (t3 - t4) ./ (E0.^3);
        Hvar = H2 - 2 * (D1.^2 + E1 ./ E0 .* H1);
         % trace
%         joint_var = diag(1./joint_precision(idx));
%         for i = 1 : N
%             xni = x(i,idx);
%             tr(i, d) = xni * joint_var * xni'; %trace(xni' * xni * joint_var);
%         end
        joint_var = 1./ joint_precision(idx);
        tr(:,d) = x(:, idx).^2 * joint_var';
        if iter > cep_iter
            new_joint_mean = E1 ./ E0 + 0.5 * H1 .* tr(:,d);
            new_joint_var = E2 ./ E0 - (E1 ./ E0).^2 + 0.5 * Hvar .* tr(:,d);
        else
            new_joint_mean = E1 ./ E0 ;
            new_joint_var = E2 ./ E0 - (E1 ./ E0).^2 ;
        end

        new_joint_precision = 1 ./ new_joint_var;
        
        new_precision(:,d) = new_joint_precision - q_precision(:,d);
        new_prod(:,d) = new_joint_precision .* new_joint_mean - q_precision(:,d) .* q_mean(:,d);
%         new_mean(:,d) = 1 ./ new_precision(:,d) .* (new_joint_precision .* new_joint_mean - q_precision(:,d) .* q_mean(:,d));
        % correction
        tmp = find(new_precision(:,d) <=0 | isnan(new_precision(:,d)) | isinf(new_precision(:,d)));
        new_precision(tmp, d) = precision_list(tmp, d);
        new_prod(tmp,d) = prod_list(tmp,d);
%         tmp = find(isnan(new_mean(:,d)));
%         new_mean(tmp, d) = mean_list(tmp, d);

    end

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
    
    disp(sprintf('lrcep2d -- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl ,auc, KL));
    if diff < threshold || iter > max_iter 
        save('./intermediate/lrcep2.mat', 'intermediate_record');
        break;
    end
    iter = iter + 1;
end

end

