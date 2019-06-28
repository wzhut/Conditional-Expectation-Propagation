function [ mse, rmse, iter, diffs] = cpep( train, test, rank, dim)
% Expecation propagation for continuous case


% train 
N = size(train, 1);
threshold = 1e-5;
rho = 0.1;

iter = 1;
max_iter = 500;
diffs = zeros([max_iter, 1]);
last_diff = 0;
last_train_error = 0;

mean_list = zeros([rank, 1, 3, N]);
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
            q_precision(:,:,m,i) = joint_precision - precision_list(:,:,m,i);
            q_mean(:,:,m,i) = q_precision(:,:,m,i) \ (joint_precision * joint_mean - precision_list(:,:,m,i) * mean_list(:,:,m,i));
        end
    end

    q_alpha = alpha + 1 - alpha_list;
    q_beta = beta - beta_list;
    
    exx = for_multinv(q_precision) + for_multiprod(q_mean, permute(q_mean, [2 1 3 4]));
    eabc = reshape(sum(q_mean(:,:,1,:) .* q_mean(:,:,2,:) .* q_mean(:,:,3,:), 1), [N, 1]); % 1 1 N
    
    % calculate new values
    new_mean_list = zeros([rank, 1, 3, N]); % r 1 3 N
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
        new_mean_list(:,:,m,:) = t_mean; % r 1 1 N
    end
    q_tau_mean = reshape(q_alpha ./ q_beta, [1 1 1 N]); 
    
    new_precision_list = bsxfun(@times, new_precision_list, q_tau_mean); % r r 3 N
    new_mean_list = bsxfun(@times, new_mean_list, vec_y .* q_tau_mean); % r 3 N  
    new_mean_list = for_multiprod(for_multinv(new_precision_list), new_mean_list);
    
    new_alpha_list = 1.5 * ones([N, 1]);
    new_beta_list = 0.5 * (y_list - eabc).^2;

    % update
    update_mean_list = mean_list * (1 - rho) + new_mean_list * rho;
    update_precision_list = precision_list * (1 - rho) + new_precision_list * rho;
    update_alpha_list = alpha_list * (1 - rho) + new_alpha_list * rho;
    update_beta_list = beta_list * (1 - rho) + new_beta_list * rho;
    
    diff = 0;
    for i = 1 : N
        for m = 1 : 3
            id = train(i,m);
            joint_mean = joint{m}.precision(:,:,id) * joint{m}.mean(:,id) - precision_list(:,:,m,i) * mean_list(:,:,m,i) + update_precision_list(:,:,m,i) * update_mean_list(:,:,m,i);
            joint_precision = joint{m}.precision(:,:,id) - precision_list(:,:,m,i) + update_precision_list(:,:,m,i);
            joint_mean = joint_precision \ joint_mean;
            
            diff = diff + mean((joint{m}.mean(:,id) - joint_mean).^2);
            joint{m}.precision(:,:,id) = joint_precision;
            joint{m}.mean(:,id) = joint_mean;
           
        end
    end
    diffs(iter) = diff;
    
    alpha = alpha - sum(alpha_list) + sum(update_alpha_list);
    beta = beta - sum(beta_list) + sum(update_beta_list);
    precision_list = update_precision_list;
    mean_list = update_mean_list;
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

