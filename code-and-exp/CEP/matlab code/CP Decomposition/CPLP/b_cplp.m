function [auc, iter, test_aucs, time] = b_cplp(train, test, r, dim, cfg)

% train

% iteration configuration
threshold = cfg.tol;
rho = cfg.rho;
max_iter = cfg.max_iter;

test_aucs = zeros([max_iter, 1]);
train_aucs = zeros([max_iter, 1]);
time = zeros([max_iter, 1]);
% last_diff = 0;
% last_auctrain = 0;


N = size(train, 1);
M = length(dim);
joint = b_initialize(train, r, dim);

mean_list = zeros([r, 1, M, N]);
precision_list = reshape(repmat(reshape(eye(r) * 1e-6, [r*r, 1]), [N * M, 1]) , [r, r, M, N]);

q_precision = zeros(r, r, M, N);
q_mean = zeros(r, 1, M, N);


y = train(:,end);
iter = 1;
while true
    tic;
    % get q\
    for i = 1 : N
        for m = 1 : M
            id = train(i, m);
            joint_precision = joint.precision{m}(:,:,id);
            joint_mean = joint.mean{m}(:,id);
            q_precision(:,:,m,i) = joint_precision - precision_list(:,:,m,i);
            q_mean(:,:,m,i) = q_precision(:,:,m,i) \ (joint_precision * joint_mean - precision_list(:,:,m,i) * mean_list(:,:,m,i));
        end
    end

     % calculate new values
    new_mean_list = zeros([r, 1, M, N]); % r 1 3 N
    new_precision_list = zeros([r, r, M, N]); % r r 3 N

    % laplace approximation
    for i = 1 : N
%         i
        param.y = y(i);
        param.r = r;
        param.mu = cell(M, 1);
        param.precision = cell(M, 1);
        x0 = rand(r * M, 1);
        for m = 1 : M
            param.mu{m} = q_mean(:,:,m,i);
            param.precision{m} = q_precision(:,:,m,i);
            x0((m - 1) * r + 1: m * r) = q_mean(:,:,m,i);
        end
        options = [];
        options.display = 'none';
        options.Method = 'lbfgs';
%         options.DERIVATIVECHECK = 'on';
        
        func = @(x) b_fun(x, param);
        [opt_x] = minFunc(func, x0, options);
        
        delta = ones(r,1);
        for m = 1 : M
            delta = delta .* opt_x((m - 1) * r + 1: m * r);
        end
        delta = sum(delta);
        cdf = normcdf((2 * y(i) - 1) * delta);
        pdf = normpdf((2 * y(i) - 1) * delta);
        
        shared = (delta * cdf + pdf) * pdf / cdf^2;
        
        for m = 1 : M
            % hessian
            select = setdiff(1:M, m);
            tmp = ones(r,1);
            for k = 1 : M - 1
                tmp = tmp .* opt_x((select(k) - 1) * r + 1: select(k) * r);
            end
            
            mean_value = opt_x((m - 1) * r + 1 : m * r);
            
            new_precision_list(:,:,m,i) = shared * (tmp * tmp');
            % ensure precision invertible
            if rank(new_precision_list(:,:,m,i)) < r
              new_precision_list(:,:,m,i) = eye(r) * 1e-6;
            end
            precision = new_precision_list(:,:,m,i) + q_precision(:,:,m,i);  
            new_mean_list(:,:,m,i) = new_precision_list(:,:,m,i) \ (precision * mean_value - q_precision(:,:,m,i) * q_mean(:,:,m,i));
        end
    end
    
    % update
    update_precision_list = precision_list * ( 1 - rho) + new_precision_list * rho;
    update_mean_list = mean_list * (1 - rho) + new_mean_list * rho;
    time(iter) = toc;
    old_joint = joint;
    for i = 1 : N
        for m = 1 : M
            id = train(i,m);
            joint_mean = joint.precision{m}(:,:,id) * joint.mean{m}(:,id) - precision_list(:,:,m,i) * mean_list(:,:,m,i) + update_precision_list(:,:,m,i) * update_mean_list(:,:,m,i);
            joint_precision = joint.precision{m}(:,:,id) - precision_list(:,:,m,i) + update_precision_list(:,:,m,i);
            joint_mean = joint_precision \ joint_mean;
            
            joint.precision{m}(:,:,id) = joint_precision;
            joint.mean{m}(:,id) = joint_mean;
        end
    end
    
    diff = 0;
    for m = 1 : 3
%        diff = diff + sum(sum((joint{m}.mean - old_joint{m}.mean).^2));  
         diff = diff + norm(joint.mean{m} - old_joint.mean{m}) / norm(old_joint.mean{m});
    end
    diff = diff / 3;
    diffs(iter) = diff;

    % replace old values
    precision_list = update_precision_list;
    mean_list = update_mean_list;
    
    % train error
    mu = ones([r, N]);
    var = ones([r, r, N]);
    for m = 1: 3
        ids = train(:,m);
        mu = mu .* joint.mean{m}(:,ids);
        var = var .* for_multinv(joint.precision{m}(:,:,ids));
    end
    smu = sum(mu)';
    svar = reshape(sum(sum(var, 2), 1), [N, 1]);
    predicted_prob = normcdf(smu./sqrt(1 + svar));
    [~,~,~,train_auc] = perfcurve(y,predicted_prob,1);
    train_aucs(iter) = train_auc;

    % test
     Nt = size(test,1);
     mu = ones([r, Nt]);
     var = ones([r, r, Nt]);
     for m = 1: 3
            ids = test(:,m);
            mu = mu .* joint.mean{m}(:,ids);
            var = var .* for_multinv(joint.precision{m}(:,:,ids));
     end
     smu = sum(mu)';
     svar = reshape(sum(sum(var, 2), 1), [Nt, 1]);
     predicted_prob = normcdf(smu./sqrt(1 + svar));
     [~,~,~,test_auc] = perfcurve(test(:,4),predicted_prob,1);
     test_aucs(iter) = test_auc;
     
    if cfg.verbose == 1
        disp(sprintf('iter:%d  diff:%.8f train_auc:%.8f test_auc:%.8f', iter, diff, train_auc, test_auc));
    end
    
    if diff < threshold || iter > max_iter
        break;
    end
     
    if iter > 10 && mean(abs(train_aucs(iter-9: iter-1) - train_auc)) < threshold
        break;
    end

    iter = iter + 1;
end

% test
 Nt = size(test,1);
 mu = ones([r, Nt]);
 var = ones([r, r, Nt]);
 for m = 1: 3
        ids = test(:,m);
        mu = mu .* joint.mean{m}(:,ids);
        var = var .* for_multinv(joint.precision{m}(:,:,ids));
 end
 smu = sum(mu)';
 svar = reshape(sum(sum(var, 2), 1), [Nt, 1]);
 predicted_prob = normcdf(smu./sqrt(1 + svar));
 [~,~,~,auc] = perfcurve(test(:,4),predicted_prob,1);

end