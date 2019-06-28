%streaming variational inference for binary tensor data
%prior_u = {prior_mean, prior_cov}
%prior_mean = {d_1 x R, ..., d_K x R}
%prior_post = {R x R x d_1, ..., R x R x d_K}
function [post_u] = POST_zhe(prior_u, batch, opt, test)
    prior_mean = prior_u{1}; 
    prior_cov = prior_u{2};
    nmod = length(prior_mean);
    R = size(prior_mean{1}, 2);
    %pre-process batch
    %indices of unique rows in each mode
    uind = cell(nmod, 1);
    %associated training entries
    data_ind = cell(nmod, 1);
    for k=1:nmod
        [uind{k}, ~, ic] = unique(batch(:,k));
        data_ind{k} = cell(length(uind{k}),1);
        for j=1:length(uind{k})
            data_ind{k}{j} = find(ic == j);
        end
    end
    
    %init vairational inference 
    %post_u = prior_u; %we can refresh
    %start with the prior, but we can use other initialization
    post_mean = prior_mean;
    post_cov = prior_cov;
    %randomize mean
    %for k=1:nmod
    %    post_mean{k}(uind{k},:) = randn(length(uind{k}),R);
    %    post_cov{k}(:,:,uind{k}) = reshape(repmat(eye(R), [1, length(uind{k})]), [R,R,length(uind{k})]);
    %end
    y = batch(:,end);
    z = 2*y - 1;
    n = size(batch,1);
    
    %train_aucs = zeros([opt.max_iter, 1]);
    %test_aucs = zeros([opt.max_iter, 1]);    
    for iter = 1:opt.max_iter
       
        old_u = cell(nmod,1);
        for k=1:nmod
            old_u{k} = post_mean{k}(uind{k},:);
        end
        for k=1:nmod            
            t_nk = ones(n,R);
            t_nk2 = ones(R,R,n);
            other_modes = setdiff(1:nmod,k);
            for j=1:length(other_modes)
                mod = other_modes(j);
                mean_batch_u = post_mean{mod}(batch(:,mod),:);
                t_nk = t_nk.*mean_batch_u;
                t_nk2 = t_nk2.*(post_cov{mod}(:,:,batch(:,mod)) + outer_prod_rows(mean_batch_u));
            end
            %update u            
            for j=1:length(uind{k})
                uid = uind{k}(j);
                eid = data_ind{k}{j};
                post_cov{k}(:, :, uid) = inv(inv(prior_cov{k}(:,:, uid)) + sum(t_nk2(:,:,eid), 3));
                post_mean{k}(uid,:) = (post_cov{k}(:, :, uid)*(prior_cov{k}(:,:,uid)\prior_mean{k}(uid,:).' ...
                    + t_nk(eid,:)'*z(eid))).';
            end
            %update z
            mean_batch_u = post_mean{k}(batch(:,k),:);
            t_k = sum(t_nk.* mean_batch_u, 2);
            %t_k2 = t_nk2.*(post_cov{k}(:,:,batch(:,k)) + outer_prod_rows(mean_batch_u);
            z = t_k + (2*y - 1).*normpdf(t_k)./normcdf((2*y-1).*t_k);
        end
        diff = 0;
        for k=1:nmod
            diff = diff + sum(sum(abs(old_u{k} - post_mean{k}(uind{k},:))));
        end
        fprintf('iter = %d, diff = %g\n', iter, diff);

%         % train auc
%         true_val = batch(:,4);
%         train_ind = batch(:,1:3);
%         n_train = size(train_ind, 1);
%         mu = ones(n_train,R);
%         S = ones(R,R,n_train);
%         for k=1:nmod
%             mu = mu.*post_mean{k}(train_ind(:,k),:);
%             S = S.*post_cov{k}(:, :, train_ind(:,k));
%         end
%         m = sum(mu,2);
%         sq = squeeze(sum(sum(S,2),1));
%         prob = normcdf(m./sqrt(1+sq)); % prob = normcdf(u/sqrt(1+sigma^2)) here, u := m
%         [~,~,~,train_auc] = perfcurve(true_val,prob,1);
%         train_aucs(iter) = train_auc;
%         
%         % test
%         true_val = test(:,4);
%         test_ind = test(:,1:3);
%         n_test = size(test_ind,1);
%         mu = ones(n_test,R);
%         S = ones(R,R,n_test);
%         for k=1:nmod
%             mu = mu.*post_mean{k}(test_ind(:,k),:);
%             S = S.*post_cov{k}(:, :, test_ind(:,k));
%         end
%         m = sum(mu,2);
%         sq = squeeze(sum(sum(S,2),1));
%         prob = normcdf(m./sqrt(1+sq)); % prob = normcdf(u/sqrt(1+sigma^2)) here, u := m
%         [~,~,~,test_auc] = perfcurve(true_val,prob,1);
%         test_aucs(iter) = test_auc;
        
        if diff< opt.tol
            break;
        end
        
       % if iter > 10 && mean(abs(train_aucs(iter-9: iter - 1) - train_auc)) < opt.tol
       %     break;
       % end
    end
    post_u = {post_mean, post_cov};
end