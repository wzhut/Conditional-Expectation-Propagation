function [logl, KL, auc, logls, KLs, aucs, time] = prcep2(train, test, ts_mean, ts_var)

threshold = 0;%1e-5;
rho = 0.05;
y = train.y;
x = train.x;
N = size(y,1);
M = size(test.y,1);
D = size(x,2);

iter = 1;
max_iter = 5000;

logls = struct('mean', {}, 'std', {});
KLs = [];
aucs = [];
time = [];

% last_diff = 0;
last_train_logl = 0;
% initialize
mean_list = zeros(N, D);
precision_list = ones(N,D) * 1e-6;
new_precision = zeros(N,D);
new_mean = zeros(N,D);
tr = zeros(N,D);
% calculate joint
joint_precision = sum(precision_list) + 1;
joint_mean = (1 ./ joint_precision) .* sum(precision_list .* mean_list);

% start timer
tic;
while true
    % q
    q_precision = repmat(joint_precision, N, 1) - precision_list;
    tmp = repmat(joint_precision .* joint_mean, N, 1);
    q_var = 1 ./ q_precision;
    q_mean = q_var .* (tmp - precision_list .* mean_list);

    for d = 1 : D
        idx = setdiff(1:D, d);
        
        c1 = (2 * y - 1) ./ sqrt(1 + (x(:,d).^2) .* q_var(:,d));
        c2 = sum((repmat(joint_mean(idx), N, 1) .* x(:,idx)),2) + x(:,d) .* q_mean(:,d);
        
        cdfln = normcdfln(c1' .* c2');
        pdfln = mvnormpdfln(c1' .* c2');
        pdc = exp(pdfln - cdfln)';
        
        dmu = pdc .* c1 .* x(:,d);
        dsig = -0.5 * pdc .* (c1.^3) .* c2 .* (x(:,d).^2);
        
        % trace
        joint_var = diag(1./joint_precision(idx));
        for i = 1 : N
            xni = x(i,idx);
            tr(i, d) = xni * joint_var * xni'; 
        end
        
        % mean
        f = q_mean(:,d) + q_var(:,d) .* dmu;
        t1 = ((c1.^2) .* (c2.^2) - 1) .* pdc;
        t2 = 3 * c1 .* c2 .* (pdc.^2);
        t3 = 2 * pdc.^3;
        mean_tr = (t1 + t2 + t3) .* (c1.^3) .* q_var(:,d) .* x(:,d) .* tr(:,d); 
        new_joint_mean = f + 0.5 * mean_tr;
        
        % variance
        f = q_var(:,d) - (q_var(:,d).^2) .* (dmu.^2 - 2 * dsig);
        t1 = c1 .* c2 .* (3 - (c1.^2) .* (c2.^2)) .* pdc;
        t2 = (4 - 7 * (c1.^2) .* (c2.^2)) .* (pdc.^2);
        t3 = -12 * c1 .* c2 .* (pdc.^3);
        t4 = -6 * (pdc.^4);
        var_tr = (t1 + t2 + t3 + t4) .* (c1.^4) .* (q_var(:,d).^2) .* (x(:,d).^2) .* tr(:,d);
        new_joint_var = f + 0.5 * var_tr;
        
        new_joint_precision = 1 ./ new_joint_var;
        new_precision(:,d) = new_joint_precision - q_precision(:,d);
        new_mean(:,d) = 1 ./ new_precision(:,d) .* (new_joint_precision .* new_joint_mean - q_precision(:,d) .* q_mean(:,d));
        
        tmp = find(new_precision(:,d) <=0);
        new_precision(tmp, d) = 1e-6;
        new_mean(tmp, d) = mean_list(tmp ,d);
    end
    
    time(iter) = toc;
    
    precision_list = precision_list * (1 - rho) + new_precision * rho;
    old_mean = mean_list;
    mean_list = mean_list * (1 - rho) + new_mean * rho;
    
    diff = mean(sum((old_mean - mean_list).^2, 2));
    
    % calculate joint
    joint_precision = sum(precision_list) + 1;
    joint_mean = (1 ./ joint_precision) .* sum(precision_list .* mean_list);
    
    % KL divergence
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
    logl.mean = mean(tmp);
    logl.std = std(tmp);

    % record
    KLs(end+1) = KL;
    logls(end+1) = logl;
    aucs(end+1) = auc;
    
%     disp(sprintf('cep2-- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl,auc, KL)); 
    disp(sprintf('cep2-- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',iter, diff, train_logl, logl.mean ,auc, KL));
    iter = iter + 1;
    if diff < threshold || iter > max_iter %|| abs(last_train_logl - train_logl) < threshold* 1e-3
        break;
    end
%     last_diff = diff;
    last_train_logl = train_logl;
end
end

