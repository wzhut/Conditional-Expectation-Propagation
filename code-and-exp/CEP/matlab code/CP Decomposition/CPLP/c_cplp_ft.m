function [mse, rmse, iter, diffs] = c_cplp(train, test, r, dim)

% train

% iteration configuration
threshold = 1e-5;
rho = 0.5;
iter = 1;
max_iter = 500;
diffs = zeros([max_iter, 1]);
last_diff = 0;
last_train_error = 0;

N = size(train, 1);
M = length(dim);
joint = c_initialize(train, r, dim);

mean_list = zeros([r, 1, M, N]);
precision_list = reshape(repmat(reshape(eye(r) * 1e-6, [r*r, 1]), [N * M, 1]) , [r, r, M, N]);

q_precision = zeros(r, r, M, N);
q_mean = zeros(r, 1, M, N);

alpha_list = ones([N, 1]);
beta_list = ones([N, 1]) * 0.01;

y = train(:,end);
while true
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

    q_alpha = joint.alpha + 1 - alpha_list;
    q_beta = joint.beta - beta_list;

        % calculate new values
    new_mean_list = zeros([r, 1, M, N]); % r 1 3 N
    new_precision_list = zeros([r, r, M, N]); % r r 3 N
    new_alpha_list = zeros(N, 1);
    new_beta_list = zeros(N, 1);

    % laplace approximation
    for i = 1 : N
%         i
        param.y = y(i);
        param.r = r;
        param.mu = cell(M, 1);
        param.precision = cell(M, 1);
        param.alpha = q_alpha(i);
        param.beta = q_beta(i);
%       fix tau
        x0 = rand(r * M, 1);
        for m = 1 : M
            param.mu{m} = q_mean(:,:,m,i);
            param.precision{m} = q_precision(:,:,m,i);
            x0((m - 1) * r + 1: m * r) = joint.mean{m}(:, train(i,m));
        end
        options = [];
        options.display = 'none';
        options.Method = 'lbfgs';
%         options.DERIVATIVECHECK = 'on';
        func = @(x) c_fun(x, param);
        [opt_x] = minFunc(func, x0, options);
        
        for m = 1 : M
            % hessian
            select = setdiff(1:M, m);
            tmp = ones(r,1);
            for k = 1 : M - 1
                tmp = tmp .* opt_x((select(k) - 1) * r + 1: select(k) * r);
            end
           
            new_precision_list(:,:,m,i) = q_alpha(i) / q_beta(i) * (tmp * tmp');
%             exp(opt_x(end))
            mean_value = opt_x((m - 1) * r + 1 : m * r);
            
            % ensure precision difinite positive
%             [~, p] = chol(new_precision_list(:,:,m,i));
            if rank(new_precision_list(:,:,m,i)) < r
              new_precision_list(:,:,m,i) = eye(r) * 1e-6;
            end
            precision = new_precision_list(:,:,m,i) + q_precision(:,:,m,i);
%               new_mean_list(:,:,m,i) = rand(r,1);
%             else
%               
%             end  
%             new_mean_list(:,:,m,i) = opt_x((m - 1) * r + 1: m * r);
            new_mean_list(:,:,m,i) = new_precision_list(:,:,m,i) \ (precision * mean_value - q_precision(:,:,m,i) * q_mean(:,:,m,i));
        end
        
        new_alpha_list(i) = 1.5;
        abc = ones(r,1);
        for m = 1 : M
           abc = abc .* q_mean(:,:,m,i); 
        end
        new_beta_list(i) = 0.5 * (y(i) - sum(abc)).^2;
    end
    
    % update
    update_precision_list = precision_list * ( 1 - rho) + new_precision_list * rho;
    update_mean_list = mean_list * (1 - rho) + new_mean_list * rho;
    update_alpha_list = alpha_list * (1 - rho) + new_alpha_list * rho;
    update_beta_list = beta_list * (1 - rho) + new_beta_list * rho;
    
    diff = 0;
    for i = 1 : N
        for m = 1 : M
            id = train(i,m);
            joint_mean = joint.precision{m}(:,:,id) * joint.mean{m}(:,id) - precision_list(:,:,m,i) * mean_list(:,:,m,i) + update_precision_list(:,:,m,i) * update_mean_list(:,:,m,i);
            joint_precision = joint.precision{m}(:,:,id) - precision_list(:,:,m,i) + update_precision_list(:,:,m,i);
            joint_mean = joint_precision \ joint_mean;
            
            diff = diff + mean((joint.mean{m}(:,id) - joint_mean).^2);
            joint.precision{m}(:,:,id) = joint_precision;
            joint.mean{m}(:,id) = joint_mean;
        end
    end
    diffs(iter) = diff;
    
    joint.alpha = joint.alpha - sum(alpha_list) + sum(update_alpha_list);
    joint.beta = joint.beta - sum(beta_list) + sum(update_beta_list);
    % replace old values
    precision_list = update_precision_list;
    mean_list = update_mean_list;
    alpha_list = update_alpha_list;
    beta_list = update_beta_list;
    
     % train error
    predicted = ones([r,N]);
    for m = 1: M
        ids = train(:,m);
        predicted = predicted .* joint.mean{m}(:,ids);
    end
    train_error = mean((y - sum(predicted)').^2);
    % test error
    Nt = size(test,1);

    predicted = ones([r,Nt]);
    for m = 1: M
            ids = test(:,m);
            predicted = predicted .* joint.mean{m}(:,ids);
    end
    test_error = mean((test(:,4) - sum(predicted)').^2);
    
    disp(sprintf('iter:%d  diff:%.8f train_error:%.8f test_error:%.8f',iter,diff,train_error,test_error));
    if diff < threshold || abs(last_train_error - train_error) < threshold
        mse = test_error;
        rmse = sqrt(test_error);
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