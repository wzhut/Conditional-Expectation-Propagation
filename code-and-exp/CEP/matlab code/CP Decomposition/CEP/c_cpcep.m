function [ mse, rmse, iter, diffs, test_rmses, time,  joint] = c_cpcep( train, test, rank, dim, cfg)
% train 

threshold = cfg.tol;
rho = cfg.rho;
max_iter = cfg.max_iter;

train_rmses = zeros([max_iter, 1]);
test_rmses = zeros([max_iter, 1]);
time = zeros([max_iter, 1]);

diffs = zeros([max_iter, 1]);

N = size(train, 1);
precision_list = repmat(eye(rank) * 1e-6, [1, 1, 3, N]);
prod_list = zeros([rank, 1, 3, N]);

alpha_list = ones([N, 1]);
beta_list = ones([N, 1]) * 0.01;

y_list = train(:,4);
vec_y = reshape(y_list, [1 1 1 N]);

% q\
q_mean = zeros([rank, 1, 3, N]);
q_precision = zeros([rank, rank, 3, N]);

% initialize
[joint, alpha, beta] = c_initialize(train, dim, rank);

iter = 1;
while true
    % calculate q\
    tic;
    for i = 1 : N
        for m = 1 : 3
            id = train(i,m);
            joint_precision = joint{m}.precision(:,:,id);
            joint_mean = joint{m}.mean(:,id);
            q_precision(:,:,m,i) = joint_precision; 
            q_mean(:,:,m,i) =joint_mean; 
        end
    end

    q_alpha = ones([N,1]) * alpha;
    q_beta = ones([N,1]) * beta;
    
    exx = for_multinv(q_precision) + for_multiprod(q_mean, permute(q_mean, [2 1 3 4]));
    eabc = reshape(sum(q_mean(:,:,1,:) .* q_mean(:,:,2,:) .* q_mean(:,:,3,:), 1), [N, 1]); % 1 1 N
    
    % calculate new values
    new_prod_list = zeros([rank, 1, 3, N]);
    new_precision_list = zeros([rank, rank, 3, N]); % r r 3 N
    
    for m = 1: 3
        t_precision = ones([rank, rank, 1, N]);
        t_mean = ones([rank, 1, 1, N]);
        index = setdiff(1:3, m);
        for k = 1 : length(index)
            t_precision = t_precision .* exx(:,:,index(k),:);
            t_mean = t_mean .* q_mean(:,:,index(k),:);
        end
        new_precision_list(:,:,m,:) = t_precision; % r r N
        new_prod_list(:,:,m,:) = t_mean; % r 1 1 N
    end
    q_tau_mean = reshape(q_alpha ./ q_beta, [1 1 1 N]); 
    
    new_precision_list = bsxfun(@times, new_precision_list, q_tau_mean); % r r 3 N
    new_prod_list = bsxfun(@times, new_prod_list, vec_y .* q_tau_mean); % r 3 N  
   
    new_alpha_list = 1.5 * ones([N, 1]);
    new_beta_list = 0.5 * (y_list - eabc).^2;

    % update  
    update_prod_list = prod_list * (1 - rho) + new_prod_list * rho;
    update_precision_list = precision_list * (1 - rho) + new_precision_list * rho;
    update_alpha_list = alpha_list * (1 - rho) + new_alpha_list * rho;
    update_beta_list = beta_list * (1 - rho) + new_beta_list * rho;
    
    time(iter) = toc;
    
    old_joint = joint;
    % calculate joint
    for i = 1 : N
        for m = 1 : 3
            id = train(i,m);
            joint_prod = joint{m}.precision(:,:,id) * joint{m}.mean(:,id) - prod_list(:,:,m,i) + update_prod_list(:,:,m,i);
            joint_precision = joint{m}.precision(:,:,id) - precision_list(:,:,m,i) + update_precision_list(:,:,m,i);
            joint_mean = joint_precision \ joint_prod;
            joint{m}.precision(:,:,id) = joint_precision;
            joint{m}.mean(:,id) = joint_mean;
        end
    end
    
    % calculate diff
    diff = 0;
    for m = 1 : 3
       diff = diff + sum(sum((joint{m}.mean - old_joint{m}.mean).^2));  
    end
    diffs(iter) = diff;
    
    % update
    alpha = alpha - sum(alpha_list) + sum(update_alpha_list);
    beta = beta - sum(beta_list) + sum(update_beta_list);
    precision_list = update_precision_list;
    prod_list = update_prod_list;
    alpha_list = update_alpha_list;
    beta_list = update_beta_list;
    
    % train error
    predicted = ones([rank,N]);
    for m = 1: 3
        ids = train(:,m);
        predicted = predicted .* joint{m}.mean(:,ids);
    end
    train_error = mean((y_list - sum(predicted)').^2);
    train_rmses(iter) = (train_error);
    % test error
    M = size(test,1);

    predicted = ones([rank,M]);
    for m = 1: 3
        ids = test(:,m);
        predicted = predicted .* joint{m}.mean(:,ids);
    end
    test_error = mean((test(:,4) - sum(predicted)').^2);
    test_rmses(iter) = (test_error);
    if cfg.verbose == 1
        disp(sprintf('iter:%d  diff:%.8f train_error:%.8f test_error:%.8f',iter,diff,train_error,test_error));
    end
    
    if diff < threshold || iter > max_iter
        break;
    end
    
    if iter > 10 && mean(abs(train_rmses(iter-9: iter-1) - train_rmses(iter))) < threshold
        break;
    end
    
    iter = iter + 1; 
end


% test
N = size(test,1);

predicted = ones([rank,N]);
for m = 1: 3
        ids = test(:,m);
        predicted = predicted .* joint{m}.mean(:,ids);
end
mse = mean((test(:,4) - sum(predicted)').^2);
rmse = sqrt(mse);

end

