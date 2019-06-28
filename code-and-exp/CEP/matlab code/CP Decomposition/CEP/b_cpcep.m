function [auc, iter, test_aucs ,time, joint] = b_cpcep(train, test, rank, dim, cfg)

% train
rho = cfg.rho;
threshold = cfg.tol;
max_iter = cfg.max_iter;

train_aucs = zeros([max_iter, 1]);
test_aucs = zeros([max_iter, 1]);
diffs = zeros([max_iter, 1]);
time = zeros([max_iter, 1]);


N = size(train, 1);
y = train(:,4);
joint = b_initialize(train, dim, rank);
v1 = 1e-6;


prod_list = zeros([rank, 1, 3, N]);
precision_list = repmat(eye(rank) * v1, [1,1,3,N]);

q_mean = zeros([rank, 1, 3, N]);
q_precision = zeros([rank, rank, 3, N]);
exx = zeros([rank, rank, 3, N]);

not_mean  = zeros([rank, 1, 3, N]);
not_precision = zeros([rank, rank, 3, N]);

new_prod_list = zeros([rank, 1, 3, N]);
new_precision_list = zeros([rank, rank, 3, N]); % r r 3 N


iter = 1;
while true 
   % prepare
   tic;
   for m = 1 : 3
       ids = train(:,m);
       q_precision(:,:,m,:) = joint{m}.precision(:,:,ids);
       q_mean(:,:,m,:) = joint{m}.mean(:,ids);
       not_precision(:,:,m,:) = joint{m}.precision(:,:,ids) - squeeze(precision_list(:,:,m,:));
       for i = 1 : N
           id = train(i,m);
           exx(:,:,m,i) = inv(q_precision(:,:,m,i)) + q_mean(:,:,m,i) * q_mean(:,:,m,i)';
           not_mean(:,:,m,i) = not_precision(:,:,m,i) \ (joint{m}.precision(:,:,id) * joint{m}.mean(:,id) - prod_list(:,:,m,i));
       end
   end

   % calculate new joint
   for m = 1:3
       index = setdiff(1:3, m);
       t = ones([rank, 1, 1, N]);
       tt = ones([rank, rank, 1, N]);
       for k = 1 : length(index)
           t = t .* q_mean(:,:, index(k),:);
           tt = tt .* exx(:,:, index(k),:);
       end
       t = squeeze(t);
       tt = squeeze(tt);

       for i = 1 : N
          trtvt = trace(not_precision(:,:,m,i) \ tt(:,:,i));
          
%           num = (2 * y(i) - 1) * t(:,i)' * not_mean(:,:,m,i);
%           den = sqrt(1 + trtvt);
%           
%           pdf = normpdf(num / den);
%           cdf = normcdf(num / den);
%           
%           dmu = pdf / cdf * (2 * y(i) - 1) / den * t(:,i);
%           dsig = pdf / cdf * (0.5 - y(i)) / den^3 * t(:,i)' * not_mean(:,:,m,i) * tt(:,:,i);
          
          c = (2 * y(i) - 1) / sqrt(1 + trtvt); 
          tmp = t(:,i)' * not_mean(:,:,m,i);
          pdf = normpdf(c * tmp);
          cdf = normcdf(c * tmp);

          dmu = pdf / cdf * c * t(:,i);
          dsig = -0.5 * pdf / cdf * c^3 * t(:,i)' * not_mean(:,:,m,i) * tt(:,:,i);
          
          A = dmu * dmu'  - 2 * dsig;
          new_joint_var = not_precision(:,:,m,i) - not_precision(:,:,m,i) * A * not_precision(:,:,m,i);
          new_joint_precision = inv(new_joint_var);
          new_joint_mean = not_mean(:,:,m,i) + not_precision(:,:,m,i) * dmu;
          new_precision_list(:,:,m,i) = new_joint_precision - not_precision(:,:,m,i);
          
          [~, p] = chol(new_precision_list(:,:,m,i));
          if p > 0
             new_precision_list(:,:,m,i) = eye(rank) * v1;
          end
          new_joint_precision = not_precision(:,:,m,i) + new_precision_list(:,:,m,i);
          new_prod_list(:,:,m,i) = new_joint_precision * new_joint_mean - not_precision(:,:,m,i) * not_mean(:,:,m,i);
          
       end
   end
   
   update_prod_list = prod_list * (1 - rho) + new_prod_list * rho;
   update_precision_list = precision_list * (1 - rho) + new_precision_list * rho;
   time(iter) = toc;
   
   % calculate joint
   old_joint = joint;
   for i = 1 : N
       for m = 1 : 3
        id = train(i,m);
        joint_prod  = joint{m}.precision(:,:,id) * joint{m}.mean(:,id) - prod_list(:,:,m,i) + update_prod_list(:,:,m,i);
        joint_precision = joint{m}.precision(:,:,id) - precision_list(:,:,m,i) + update_precision_list(:,:,m,i);
        joint_mean = joint_precision \ joint_prod;
     
        joint{m}.precision(:,:,id) = joint_precision;
        joint{m}.mean(:,id) = joint_mean;
       end
   end
   % update
   prod_list = update_prod_list;
   precision_list = update_precision_list;
   
   % calculate diff
    diff = 0;
    for m = 1 : 3
       diff = diff + sum(sum((joint{m}.mean - old_joint{m}.mean).^2));  
%          diff = diff + norm(joint{m}.mean - old_joint{m}.mean) / norm(old_joint{m}.mean);
    end
    diff = diff / 3;
    diffs(iter) = diff;
    
    
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
    train_aucs(iter) = train_auc;

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
    test_aucs(iter) = test_auc;
    
    
    if cfg.verbose == 1
        disp(sprintf('iter:%d  diff:%.10f auctrain:%.4f auctest:%.4f',iter,diff, train_auc, test_auc));
    end

   if diff < threshold || iter > max_iter
        break;
   end
   
   if iter > 10 && mean(abs(train_aucs(iter-9: iter - 1) - train_auc)) < threshold
        break;
   end
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