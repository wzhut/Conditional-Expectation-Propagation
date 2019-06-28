function [auc, iter, aucs] = b_bcpep(train, test, rank, dim)

% train
N = size(train, 1);
rho = 0.1;
threshold = 1e-5;
auc_threshold = 1e-5;
iter = 1;
max_iter = 500;
aucs = zeros([max_iter, 1]);
last_auc = 0;
last_diff = 0;


y = train(:,4);

joint = binitialize(train, dim, rank);

v1 = 1e-6;

mean_list = zeros([rank, 1, 3, N]);
precision_list = reshape(repmat(reshape(eye(rank) * v1, [rank * rank, 1]), [N * 3, 1]), [rank, rank, 3, N]);

q_mean = zeros([rank, 1, 3, N]);
q_precision = zeros([rank, rank, 3, N]);

new_mean_list = zeros([rank, 1, 3, N]); % r 1 3 N
new_precision_list = zeros([rank, rank, 3, N]); % r r 3 N
while true 
   for i = 1 : N
       for m = 1 : 3
        id = train(i, m);
        q_precision(:,:,m,i) = joint{m}.precision(:,:,id) - precision_list(:,:,m,i);
        q_mean(:,:,m,i) = q_precision(:,:,m,i) \ (joint{m}.precision(:,:,id) * joint{m}.mean(:,id) - precision_list(:,:,m,i) * mean_list(:,:,m,i)); 
       end
   end
   
   q_var = for_multinv(q_precision);
   q_mean_t = permute(q_mean, [2 1 3 4]);
   exx = q_var + for_multiprod(q_mean, q_mean_t);
   
   for m = 1:3
       index = setdiff(1:3, m);
       t = ones([rank, 1, 1, N]);
       tt = ones([rank, rank, 1, N]);
       for k = 1 : length(index)
           t = t .* q_mean(:,:, index(k),:);
           tt = tt .* exx(:,:, index(k),:);
       end
       t_t = permute(t, [2 1 3 4]);
       trtvt = zeros([N, 1]);
       tmp = for_multiprod(q_var(:,:,m,:), tt);
       
       for i = 1 : N
           trtvt(i) = trace(tmp(:,:,1,i));
       end
       
       num = (2 * y - 1) .* reshape(for_multiprod(t_t, q_mean(:,:,m,:)), [N, 1]);
       den = sqrt(1 + trtvt);
       
       cdf = normcdf(num ./ den);
       pdf = normpdf(num ./ den);
       
       dmu = bsxfun(@times, reshape(pdf ./ cdf .* (2 * y - 1) ./ den, [1, 1, 1, N]), t);
       dsig = bsxfun(@times, reshape(pdf ./ cdf .* (0.5 - y) ./ den.^3, [1, 1, 1, N]), for_multiprod(for_multiprod(t_t, q_mean(:,:,m,:)), tt));
       
       A = for_multiprod(dmu, permute(dmu, [2 1 3 4])) - 2 * dsig;
       new_joint_precision = for_multinv(q_var(:,:,m,:)  - for_multiprod(q_var(:,:,m,:), for_multiprod(A, q_var(:,:,m,:))));
       new_precision_list(:,:,m,:) = new_joint_precision - q_precision(:,:,m,:);
       
%        for k= 1 : N
%            [~, p] = chol(new_precision_list(:,:,m,k));
%            if p > 0
%               new_precision_list(:,:,m,k) = eye(rank) * v1;
%               new_joint_precision = q_precision(:,:,m,k) + new_precision_list(:,:,m,k);
%            end
%         end
       
       
       new_joint_product = for_multiprod(new_joint_precision, q_mean(:,:,m,:) + for_multiprod(q_var(:,:,m,:), dmu));
       new_mean_list(:,:,m,:) = for_multiprod(for_multinv(new_precision_list(:,:,m,:)), new_joint_product - for_multiprod(q_precision(:,:,m,:), q_mean(:,:,m,:)));
   end
   
   update_mean_list = mean_list * (1 - rho) + new_mean_list * rho;
   update_precision_list = precision_list * (1 - rho) + new_precision_list * rho;
   
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
   

    
       % train error
    mu = ones([rank, N]);
    var = ones([rank, rank, N]);
    for m = 1: 3
        ids = train(:,m);
        mu = mu .* joint{m}.mean(:,ids);
        var = var .* for_multinv(joint{m}.precision(:,:,ids));
    end
    smu = sum(mu)';
    svar = reshape(sum(sum(var, 2), 1), [N, 1]);
    predicted_prob = normcdf(smu./sqrt(1 + svar));
    [~,~,~,train_auc] = perfcurve(y,predicted_prob,1);
    
    % test
     M = size(test,1);
     mu = ones([rank, M]);
     var = ones([rank, rank, M]);
     for m = 1: 3
            ids = test(:,m);
            mu = mu .* joint{m}.mean(:,ids);
            var = var .* for_multinv(joint{m}.precision(:,:,ids));
     end
     smu = sum(mu)';
     svar = reshape(sum(sum(var, 2), 1), [M, 1]);
     predicted_prob = normcdf(smu./sqrt(1 + svar));
     [~,~,~,test_auc] = perfcurve(test(:,4),predicted_prob,1);
    
    if iter > max_iter
        break;
    end
    
    
   disp(sprintf('iter:%d  diff:%.10f auctrain:%.4f auctest:%.4f',iter,diff, train_auc, test_auc));
   
   if diff < threshold || abs(last_auc - train_auc) < auc_threshold
        break;
   end
   last_auc = train_auc;
   iter = iter + 1;
   
end
% % test
 M = size(test,1);
 mu = ones([rank, M]);
 var = ones([rank, rank, M]);
 for m = 1: 3
        ids = test(:,m);
        mu = mu .* joint{m}.mean(:,ids);
        var = var .* for_multinv(joint{m}.precision(:,:,ids));
 end
 smu = sum(mu)';
 svar = reshape(sum(sum(var, 2), 1), [M, 1]);
 predicted_prob = normcdf(smu./sqrt(1 + svar));
 [~,~,~,auc] = perfcurve(test(:,4),predicted_prob,1);

end