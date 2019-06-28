function [mse, rmse, iter, diffs] = sc_cplp(train, test, rank, dim)

% train

% iteration configuration
threshold = 0;
rho = 0.9;
iter = 1;
max_iter = 500;
diffs = zeros([max_iter, 1]);
last_diff = 0;
last_train_error = 0;

N = size(train, 1);
M = length(dim);
joint = c_initialize(train, rank, dim);

mean_list = zeros([rank, 1, M, N]);
precision_list = reshape(repmat(reshape(eye(rank) * 1e-6, [rank*rank, 1]), [N * M, 1]) , [rank, rank, M, N]);

alpha_list = ones([N, 1]);
beta_list = ones([N, 1]) * 0.01;

y = train(:,end);
new_mean = zeros([rank, 1, M]);
new_precision = zeros([rank, rank, M]);

q_mean = zeros([rank, 1, M]);
q_precision = zeros([rank, rank, M]);
    
while true
    diff = 0;
    a=0;
    b=0;
    c=0;
    % laplace approximation
    for i = 1 : N
%         q
        for m = 1 : M
            id = train(i, m);
            joint_precision = joint.precision{m}(:,:,id);
            joint_mean = joint.mean{m}(:,id);
            q_precision(:,:,m) = joint_precision - precision_list(:,:,m,i);
            q_mean(:,:,m) = q_precision(:,:,m) \ (joint_precision * joint_mean - precision_list(:,:,m,i) * mean_list(:,:,m,i));
        end
        
        q_alpha = joint.alpha + 1 - alpha_list(i);
        q_beta = joint.beta - beta_list(i);
        
        param.y = y(i);
        param.rank = rank;
        param.mu = cell(M, 1);
        param.precision = cell(M, 1);
        param.alpha = q_alpha;
        param.beta = q_beta;
        x0 = rand(rank * M + 1, 1);
%          x0(end) = -6;
        for m = 1 : M
            param.mu{m} = q_mean(:,:,m);
            param.precision{m} = q_precision(:,:,m);
%             x0((m - 1) * rank + 1: m * rank) = joint.mean{m}(:,train(i,m));
        end
        options = [];
        options.display = 'none';
        options.Method = 'lbfgs';
        func = @(x) c_fun(x, param);
        [opt_x] = minFunc(func, x0, options);
        
        for m = 1 : M
            % hessian
            select = setdiff(1:M, m);
            tmp = ones(rank,1);
            for k = 1 : M - 1
                tmp = tmp .* opt_x((select(k) - 1) * rank + 1: select(k) * rank);
            end
            precision = exp(opt_x(end)) * (tmp * tmp') + q_precision(:,:,m);

            mean_value = opt_x((m - 1) * rank + 1 : m * rank);
            new_precision(:,:,m) = precision - q_precision(:,:,m);
            % ensure precision difinite positive
            [~, p] = chol(new_precision(:,:,m));
            if p > 0
              new_precision(:,:,m) = eye(rank) * 1e-6;
%               new_mean(:,:,m) = rand(rank,1);
              c = c + 1;
            else
              
            end     
            new_mean(:,:,m) = new_precision(:,:,m) \ (precision * mean_value - q_precision(:,:,m) * q_mean(:,:,m));
        end
%       log tau
        mean_lt = opt_x(end);
        
        for m = 1 : M
            tmp = ones(rank,1);
            tmp = tmp .* opt_x((m - 1) * rank + 1: m * rank);
        end
        var_lt = 0.5 * (sum(tmp) - y(i)) ^ 2 + exp(mean_lt) * q_beta;
        mean_t = exp(mean_lt + var_lt / 2);
        var_t = exp(2 * mean_lt + var_lt) * (exp(var_lt) - 1) + realmin;
        
        
        joint_beta = mean_t / var_t;
        joint_alpha = joint_beta * mean_t;
        new_alpha = joint_alpha - q_alpha + 1;
        if new_alpha < 1
            new_alpha = 1;
            a  = a + 1;
        end
        new_beta = joint_beta - q_beta;
        if new_beta < 0
            new_beta = 0.01;
            b = b + 1;
        end
        % update
        update_precision = precision_list(:,:,:,i) * ( 1 - rho) + new_precision * rho;
        update_mean = mean_list(:,:,:,i) * (1 - rho) + new_mean * rho;
        update_alpha = alpha_list(i) * (1 - rho) + new_alpha * rho;
        update_beta = beta_list(i) * (1 - rho) + new_beta * rho;
        
        for m = 1 : M
            id = train(i,m);
            joint_mean = joint.precision{m}(:,:,id) * joint.mean{m}(:,id) - precision_list(:,:,m,i) * mean_list(:,:,m,i) + update_precision(:,:,m) * update_mean(:,:,m);
            joint_precision = joint.precision{m}(:,:,id) - precision_list(:,:,m,i) + update_precision(:,:,m);
            joint_mean = joint_precision \ joint_mean;
            
            diff = diff + mean((joint.mean{m}(:,id) - joint_mean).^2);
            joint.precision{m}(:,:,id) = joint_precision;
            joint.mean{m}(:,id) = joint_mean;
        end
        
        joint.alpha = joint.alpha - alpha_list(i) + update_alpha;
        joint.beta = joint.beta - beta_list(i) + update_beta;
        
            % replace old values
        precision_list(:,:,:,i) = update_precision;
        mean_list(:,:,:,i) = update_mean;
        alpha_list(i) = update_alpha;
        beta_list(i) = update_beta;
    end
    
    diffs(iter) = diff;
        
     % train error
    predicted = ones([rank,N]);
    for m = 1: M
        ids = train(:,m);
        predicted = predicted .* joint.mean{m}(:,ids);
    end
    train_error = mean((y - sum(predicted)').^2);
    % test error
    Nt = size(test,1);

    predicted = ones([rank,Nt]);
    for m = 1: M
            ids = test(:,m);
            predicted = predicted .* joint.mean{m}(:,ids);
    end
    test_error = mean((test(:,4) - sum(predicted)').^2);
    
    disp(sprintf('iter:%d  diff:%.8f train_error:%.8f test_error:%.8f',iter,diff,train_error,test_error));
    a
    b
    c
    if diff < threshold || abs(last_train_error - train_error) < threshold
        break;
    end

    if iter > max_iter
        break;
    end
    
    iter = iter + 1;
    last_diff = diff;
    last_train_error = train_error;
end

end