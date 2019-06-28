function [ mse, rmse, iter, diffs] = cpcep( train, test, rank, dim)
% Expecation propagation for continuous case


% train 
N = size(train, 1);
threshold = 0;
rho = 0.1;

iter = 1;
max_iter = 500;
diffs = zeros([max_iter, 1]);
last_diff = 0;
last_train_error = 0;

% mean_list = zeros([rank, 1, 3, N]);
prod_list = zeros([rank, 1, 3, N]);
precision_list = reshape(repmat(reshape(eye(rank) * 1e-6, [rank*rank, 1]), [N * 3, 1]) , [rank, rank, 3, N]);

alpha_list = ones([N, 1]);
beta_list = ones([N, 1]) * 0.01;

y_list = train(:,4);
vec_y = reshape(y_list, [1 1 1 N]);

% q\
q_mean = zeros([rank, 1, 3, N]);
q_precision = zeros([rank, rank, 3, N]);
% initialize
[joint, alpha, beta] = initialize(train, dim, rank);


while true
    % calculate q\
    for i = 1 : N
        for m = 1 : 3
            id = train(i,m);
            joint_precision = joint{m}.precision(:,:,id);
            joint_mean = joint{m}.mean(:,id);
            q_precision(:,:,m,i) = joint_precision; % - precision_list(:,:,m,i);
            q_mean(:,:,m,i) =joint_mean;  % q_precision(:,:,m,i) \ (joint_precision * joint_mean - precision_list(:,:,m,i) * mean_list(:,:,m,i));
        end
    end
    
    q_alpha = ones([N,1]) * alpha;
    q_beta = ones([N,1]) * beta;
    
%     exx = for_multinv(q_precision) + for_multiprod(q_mean, permute(q_mean, [2 1 3 4]));
    eabc = reshape(sum(q_mean(:,:,1,:) .* q_mean(:,:,2,:) .* q_mean(:,:,3,:), 1), [N, 1]); % 1 1 N
    
    
    for m = 1: 3
        t_prod = ones([rank, 1, 1, N]);
        index = setdiff(1:3, m);
        for k = 1 : length(index)
            t_prod = t_prod .* q_mean(:,:,index(k),:);
        end
        t_precision = for_multiprod(t_prod, permute(t_prod, [2 1 3 4]));
    end
    q_tau_mean = reshape(q_alpha ./ q_beta, [1 1 1 N]); 
    
    t_precision = bsxfun(@times, t_precision, q_tau_mean); % r r 3 N
    t_prod = bsxfun(@times, t_prod, vec_y .* q_tau_mean); % r 3 N  

    
    new_alpha_list = 1.5 * ones([N, 1]);
    new_beta_list = 0.5 * (y_list - eabc).^2;

    % update
    update_prod_list = prod_list * (1 - rho^2) + t_prod * rho^2;
    update_precision_list = precision_list * (1 - rho) + t_precision * rho;
    
    update_alpha_list = alpha_list * (1 - rho) + new_alpha_list * rho;
    update_beta_list = beta_list * (1 - rho) + new_beta_list * rho;
    
    diff = 0;
    old_joint = joint;
    for i = 1 : N
        for m = 1 : 3
            id = train(i,m);
            joint_mean = joint{m}.precision(:,:,id) * joint{m}.mean(:,id) - prod_list(:,:,m,i) + update_prod_list(:,:,m,i);
            joint_precision = joint{m}.precision(:,:,id) - precision_list(:,:,m,i) + update_precision_list(:,:,m,i);
            joint_mean = joint_precision \ joint_mean;
            
             diff = diff + mean((joint{m}.mean(:,id) - joint_mean).^2);
            joint{m}.precision(:,:,id) = joint_precision;
            joint{m}.mean(:,id) = joint_mean;
           
        end
    end
    
%     for m = 1 : 3
%         diff = diff + mean(sum((joint{m}.mean - old_joint{m}.mean).^2));
%     end
    
    diffs(iter) = diff;
    
    alpha = alpha - sum(alpha_list) + sum(update_alpha_list);
    beta = beta - sum(beta_list) + sum(update_beta_list);
    precision_list = update_precision_list;
    prod_list = update_prod_list;
%     mean_list = update_mean_list;
    alpha_list = update_alpha_list;
    beta_list = update_beta_list;
    
    
    % train error
    predicted = ones([rank,N]);
    for m = 1: 3
        ids = train(:,m);
        predicted = predicted .* joint{m}.mean(:,ids);
    end
    train_error = mean((y_list - sum(predicted)').^2);
    % test error
    M = size(test,1);

    predicted = ones([rank,M]);
    for m = 1: 3
            ids = test(:,m);
            predicted = predicted .* joint{m}.mean(:,ids);
    end
    test_error = mean((test(:,4) - sum(predicted)').^2);
    
    if diff < threshold || abs(last_train_error - train_error) < threshold
        break;
    end
%     if diff < threshold
%         break;
%     end
    if iter > max_iter
        break;
    end
    disp(sprintf('iter:%d  diff:%.8f train_error:%.8f test_error:%.8f',iter,diff,train_error,test_error));
    iter = iter + 1;
    last_diff = diff;
    last_train_error = train_error;
    
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

