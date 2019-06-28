function [U_invS_g, U_invSMean_g] = b_cpadf_v2(train, test, dim, nvec, batch_size, cfg)

aucs=[];
v1 = 1e-6;
% train
num_batch = ceil(size(train,1) / batch_size);

[~, nmod] = size(train);

%global
U_invS_g = cell(nmod, 1);
U_invSMean_g = cell(nmod, 1);
U_invSMean_prior = cell(nmod, 1);

for nbatch = 1 : num_batch
    fprintf('batch: %g\n', nbatch);
    if nbatch < num_batch
        cur_batch = train((nbatch - 1) * batch_size + 1 : nbatch * batch_size, :);
    else
        cur_batch = train((nbatch - 1) * batch_size + 1 : end, :);
    end
    % sample number
    
    max_iter = cfg.max_iter;
    train_aucs = zeros([max_iter, 1]);
    test_aucs = zeros([max_iter, 1]);
%     diffs = zeros([max_iter, 1]);
%     time = zeros([max_iter, 1]);

    [N, nmod] = size(cur_batch);
    nmod = nmod - 1;
    %nvec = max(train(:,1:end-1));
    y = 2*cur_batch(:,end)-1;

    %indices of unique rows in each mode
    uind = cell(nmod, 1);
    data_ind = cell(nmod, 1);
    for k=1:nmod
        [uind{k}, ~, ic] = unique(cur_batch(:,k));
        data_ind{k} = cell(length(uind{k}),1);
        for j=1:length(uind{k})
            %mode k, j-th entity in uind{k} appears in which entries
            data_ind{k}{j} = find(ic == j);
        end
    end

    %each entry
    U_invS = cell(nmod, 1);
    U_invSMean = cell(nmod, 1);
 
    U_invS_not = cell(nmod,1);
    U_invSMean_not = cell(nmod,1);
    infty = 1e6;
    for k=1:nmod
        U_invS{k} = repmat(eye(dim)*1/infty, [1, 1, N]);
        U_invSMean{k} = zeros(dim, 1, N);
        U_invS_not{k} = zeros(dim, dim, N);
        U_invSMean_not{k} = zeros(dim, 1, N);
        %start with prior, must be randomly init
        if nbatch == 1
            U_invS_g{k} = repmat(eye(dim), [1, 1, nvec(k)]);
            U_invSMean_prior{k} = rand(dim, 1, nvec(k));
            U_invSMean_g{k} =  U_invSMean_prior{k};
            U_invS_prior{k} = repmat(eye(dim), [1, 1, nvec(k)]);
        else
            U_invS_prior{k} = U_invS_g{k};
            U_invSMean_prior{k} = U_invSMean_g{k};
        end
    end
    U_cov_not = cell(nmod,1);
    U_mean_not = cell(nmod, 1);
    U_invS_new = cell(nmod, 1);
    U_invSMean_new = cell(nmod, 1);
    %get the global
    for k=1:nmod
        for j=1:length(uind{k})
            U_invS_g{k}(:,:,uind{k}(j)) = sum(U_invS{k}(:,:,data_ind{k}{j}), 3) + U_invS_prior{k}(:,:,uind{k}(j));
            U_invSMean_g{k}(:, :, uind{k}(j)) = sum(U_invSMean{k}(:, :, data_ind{k}{j}), 3) + U_invSMean_prior{k}(:, :, uind{k}(j));        
        end
    end

    for iter = 1:cfg.max_iter
%         tic;
        u_old = U_invSMean_g;
        for k=1:nmod
            %calibrating 
            U_invS_not{k} = U_invS_g{k}(:, :, cur_batch(:,k)) - U_invS{k};
            U_invSMean_not{k} = U_invSMean_g{k}(:, :, cur_batch(:,k)) - U_invSMean{k};
            U_cov_not{k} = multinv(U_invS_not{k});
            U_mean_not{k} = mtimesx(U_cov_not{k}, U_invSMean_not{k}); 
        end
        for k=1:nmod
            z_nk = ones(dim, 1, N);
            z_nk2 = ones(dim, dim, N);
            other_modes = setdiff(1:nmod,k);
            for j=1:length(other_modes)
                mode = other_modes(j);
                z_nk = z_nk.*U_mean_not{mode};
                z_nk2 = z_nk2.*(U_cov_not{mode} + mtimesx(U_mean_not{mode},U_mean_not{mode}, 't'));
            end
            
            mu = y.*reshape(mtimesx(U_mean_not{k},'t', z_nk),[N,1]); 
            s = sum(reshape(z_nk2.*U_cov_not{k}, [dim*dim, N]),1)';
            z = mu./sqrt(1+s);
            yr = y./sqrt(1+s);         
            N_phi = exp(mvnormpdfln(z')' - normcdfln(z));
            Dm = repmat(reshape(N_phi.*yr,[1,1,N]),[dim,1]).*z_nk;
            DS = repmat(reshape(-0.5*N_phi.*(yr.^2).*z,[1,1,N]),[dim,dim]).*z_nk2;
            A = mtimesx(Dm, Dm,'t') - 2*DS;
            mean_new = U_mean_not{k} + mtimesx(U_cov_not{k},Dm);
            cov_new = U_cov_not{k} - mtimesx(mtimesx(U_cov_not{k},A), U_cov_not{k});
            U_invS_new{k} = multinv(cov_new);
            U_invSMean_new{k} = mtimesx(U_invS_new{k}, mean_new);
        end
        %damping & update
        diff = 0;
        for k=1:nmod
            U_invS{k} = U_invS{k}*(1-cfg.rho) + (U_invS_new{k} - U_invS_not{k}) *cfg.rho;
            for i = 1 : N
                [~, p] = chol(U_invS{k}(:,:,i));
                if p > 0
                    U_invS{k}(:,:,i) = eye(dim);
                end
            end
            U_invSMean{k} = U_invSMean{k}*(1-cfg.rho) + (U_invSMean_new{k} - U_invSMean_not{k}) *cfg.rho;

            for j=1:length(uind{k})
                U_invS_g{k}(:,:,uind{k}(j)) = sum(U_invS{k}(:,:,data_ind{k}{j}), 3) +  U_invS_prior{k}(:,:,uind{k}(j));%eye(dim);
                U_invSMean_g{k}(:, :, uind{k}(j)) = sum(U_invSMean{k}(:, :, data_ind{k}{j}), 3) + U_invSMean_prior{k}(:, :, uind{k}(j));        
            end
            diff = diff + sum(abs(vec(U_invSMean_g{k}) - vec(u_old{k})));
            %diff = diff + norm(vec(U_invSMean_g{k}) - vec(u_old{k}))/norm(vec(u_old{k}));
        end
%         time(iter) = toc;
        diff = diff/nmod;
%         fprintf('iter = %d, diff = %g\n', iter, diff);
        %if mod(iter, 10)==0
        
%         pred = ones(dim, 1, size(cur_batch,1));
%         pred_var = ones(dim, dim, size(cur_batch,1));
%         U_cov_g = cell(nmod,1);
%         U_mean_g = cell(nmod,1);
% 
%         for k=1:nmod
%             U_cov_g{k} = multinv(U_invS_g{k});
%             U_mean_g{k} = mtimesx(U_cov_g{k}, U_invSMean_g{k});                
%             pred = pred.*U_mean_g{k}(:,:,cur_batch(:,k));
%             pred_var = pred_var.*U_cov_g{k}(:,:,cur_batch(:,k));
%         end
%         pred = sum(reshape(pred,[dim,size(cur_batch,1)]),1)';
%         pred_var = sum(reshape(pred_var, [dim*dim, size(cur_batch,1)]),1)';
%         prob = normcdf(pred./sqrt(1 + pred_var));
%         %prob = pred;
%         [~,~,~,auc] = perfcurve(cur_batch(:,end),prob,1);
%         train_aucs(iter) = auc;
%         
%         pred = ones(dim, 1, size(test,1));
%         pred_var = ones(dim, dim, size(test,1));
%         U_cov_g = cell(nmod,1);
%         U_mean_g = cell(nmod,1);
% 
%         for k=1:nmod
%             U_cov_g{k} = multinv(U_invS_g{k});
%             U_mean_g{k} = mtimesx(U_cov_g{k}, U_invSMean_g{k});                
%             pred = pred.*U_mean_g{k}(:,:,test(:,k));
%             pred_var = pred_var.*U_cov_g{k}(:,:,test(:,k));
%         end
%         pred = sum(reshape(pred,[dim,size(test,1)]),1)';
%         pred_var = sum(reshape(pred_var, [dim*dim, size(test,1)]),1)';
%         prob = normcdf(pred./sqrt(1 + pred_var));
%         %prob = pred;
%         [~,~,~,auc] = perfcurve(test(:,end),prob,1);
%         test_aucs(iter) = auc;
        
        if cfg.verbose == 1
                    pred = ones(dim, 1, size(test,1));
        pred_var = ones(dim, dim, size(test,1));
        U_cov_g = cell(nmod,1);
        U_mean_g = cell(nmod,1);

        for k=1:nmod
            U_cov_g{k} = multinv(U_invS_g{k});
            U_mean_g{k} = mtimesx(U_cov_g{k}, U_invSMean_g{k});                
            pred = pred.*U_mean_g{k}(:,:,test(:,k));
            pred_var = pred_var.*U_cov_g{k}(:,:,test(:,k));
        end
        pred = sum(reshape(pred,[dim,size(test,1)]),1)';
        pred_var = sum(reshape(pred_var, [dim*dim, size(test,1)]),1)';
        prob = normcdf(pred./sqrt(1 + pred_var));
        %prob = pred;
        [~,~,~,auc] = perfcurve(test(:,end),prob,1);
        test_aucs(iter) = auc;
            fprintf('iter = %d, diff = %g, train_auc = %g, test_auc = %g\n', iter, diff, train_aucs(iter), test_aucs(iter));
        end
        
        if diff < cfg.tol
            break;
        end
        
%         if iter > 10 && mean(abs(train_aucs(iter-9: iter - 1) - train_aucs(iter))) < cfg.tol
%             break;
%         end
    end
    
end


end