function [auc, iter, aucs, joint] = b_cpadf(train, test, rank, dim, batch_size, cfg)

aucs=[];
v1 = 1e-6;
% train
num_batch = ceil(size(train,1) / batch_size);

threshold = cfg.tol;
rho = cfg.rho;
max_iter = cfg.max_iter;

% initialize
joint = b_initialize(train, dim, rank);
N = batch_size;



for nbatch = 1 : num_batch
    if nbatch < num_batch
        cur_batch = train((nbatch - 1) * batch_size + 1 : nbatch * batch_size, :);
    else
        cur_batch = train((nbatch - 1) * batch_size + 1 : end, :);
    end
    % sample number
    N = size(cur_batch, 1);
    prod_list = zeros([rank, 1, 3, N]);
    precision_list = repmat(eye(rank) * v1, [1,1,3,N]);

    q_mean = zeros([rank, 1, 3, N]);
    q_precision = zeros([rank, rank, 3, N]);

    not_mean  = zeros([rank, 1, 3, N]);
    not_precision = zeros([rank, rank, 3, N]);

    new_prod_list = zeros([rank, 1, 3, N]);
    new_precision_list = zeros([rank, rank, 3, N]); % r r 3 N
    
    y = cur_batch(:,4);
    % joint
    for m = 1: 3
        for i = 1 : dim(m)
            num = sum(double(cur_batch(:,m) == i));
            joint_precision = num * eye(rank) * v1 + joint{m}.precision(:,:,i);
            joint_mean = joint_precision \ (joint{m}.precision(:,:,i) * joint{m}.mean(:,i));
            joint{m}.mean(:,i) = joint_mean;
            joint{m}.precision(:,:,i) = joint_precision;
        end
    end
    
    iter = 1;
    while true 
       for i = 1 : N
           for m = 1 : 3
            id = cur_batch(i, m);
            q_precision(:,:,m,i) = joint{m}.precision(:,:,id); 
            q_mean(:,:,m,i) = joint{m}.mean(:,id); 

            not_precision(:,:,m,i) = joint{m}.precision(:,:,id) - precision_list(:,:,m,i);
            not_mean(:,:,m,i) = not_precision(:,:,m,i) \ (joint{m}.precision(:,:,id) * joint{m}.mean(:,id) - prod_list(:,:,m,i)); 
           end
       end

       q_var = for_multinv(q_precision);
       not_var = for_multinv(not_precision);

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
           tmp = for_multiprod(not_var(:,:,m,:), tt);

           for i = 1 : N
               trtvt(i) = trace(tmp(:,:,1,i));
           end

           num = (2 * y - 1) .* reshape(for_multiprod(t_t, not_mean(:,:,m,:)), [N, 1]);
           den = sqrt(1 + trtvt);

           cdf = normcdf(num ./ den);
           pdf = normpdf(num ./ den);

           dmu = bsxfun(@times, reshape(pdf ./ cdf .* (2 * y - 1) ./ den, [1, 1, 1, N]), t);
           dsig = bsxfun(@times, reshape(pdf ./ cdf .* (0.5 - y) ./ den.^3, [1, 1, 1, N]), for_multiprod(for_multiprod(t_t, not_mean(:,:,m,:)), tt));

           A = for_multiprod(dmu, permute(dmu, [2 1 3 4])) - 2 * dsig;
           new_joint_precision = for_multinv(not_var(:,:,m,:)  - for_multiprod(not_var(:,:,m,:), for_multiprod(A, not_var(:,:,m,:))));
           new_precision_list(:,:,m,:) = new_joint_precision - not_precision(:,:,m,:);
           
%            for k= 1 : N
%                [~, p] = chol(new_precision_list(:,:,m,k));
%                if p > 0
%                   new_precision_list(:,:,m,k) = eye(rank) * v1;
% %                   new_prod_list(:,:,m,:) = prod_list(:,:,m,:);
%                end
%            end
%            new_joint_precision = new_precision_list(:,:,m,:) + not_precision(:,:,m,:);
           
           new_joint_product = for_multiprod(new_joint_precision, not_mean(:,:,m,:) + for_multiprod(not_var(:,:,m,:), dmu));
           new_prod_list(:,:,m,:) = new_joint_product - for_multiprod(not_precision(:,:,m,:), not_mean(:,:,m,:));
           
%            for k= 1 : N
%                [~, p] = chol(new_precision_list(:,:,m,k));
%                if p > 0
%                   new_precision_list(:,:,m,k) = eye(rank) * v1;
% %                   new_prod_list(:,:,m,:) = prod_list(:,:,m,:);
%                end
%            end
       end
       
       update_prod_list = prod_list * (1 - rho) + new_prod_list * rho;
       update_precision_list = precision_list * (1 - rho) + new_precision_list * rho;

       % calculate joint
       old_joint = joint;
       for i = 1 : N
           for m = 1 : 3
            id = cur_batch(i,m);
            joint_prod  = joint{m}.precision(:,:,id) * joint{m}.mean(:,id) - prod_list(:,:,m,i) + update_prod_list(:,:,m,i);
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
%         diffs(iter) = diff;

        % update
        prod_list = update_prod_list;
        precision_list = update_precision_list;

        if cfg.verbose == 1
            % train error
%             mu = ones([rank, N]);
%             var = ones([rank, rank, N]);
%             for m = 1: 3
%                 ids = cur_batch(:,m);
%                 mu = mu .* joint{m}.mean(:,ids);
%                 var = var .* for_multinv(joint{m}.precision(:,:,ids));
%             end
%             smu = sum(mu)';
%             svar = reshape(sum(sum(var, 2), 1), [N, 1]);
%             predicted_prob = normcdf(smu./sqrt(1 + svar));
%             [~,~,~,train_auc] = perfcurve(y, predicted_prob,1);
            train_auc = -1;

            %         test
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
            disp(sprintf('batch:%d, iter:%d  diff:%.10f auctrain:%.4f auctest:%.4f',nbatch, iter,diff, train_auc, test_auc));
        end

        if iter > max_iter || diff < threshold 
            break;
        end
       iter = iter + 1;
    end
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