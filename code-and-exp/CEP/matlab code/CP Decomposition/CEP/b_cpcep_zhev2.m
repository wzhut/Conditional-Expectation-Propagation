%CEP for binary tensor
function [U_invS_g, U_invSMean_g] = b_cpcep_zhev2( train, test, dim, nvec, cfg)
    [N, nmod] = size(train);
    nmod = nmod - 1;
    %nvec = max(train(:,1:end-1));
    y = 2*train(:,end)-1;

    %indices of unique rows in each mode
    uind = cell(nmod, 1);
    data_ind = cell(nmod, 1);
    for k=1:nmod
        [uind{k}, ~, ic] = unique(train(:,k));
        data_ind{k} = cell(length(uind{k}),1);
        for j=1:length(uind{k})
            %mode k, j-th entity in uind{k} appears in which entries
            data_ind{k}{j} = find(ic == j);
        end
    end

    %each entry
    U_invS = cell(nmod, 1);
    U_invSMean = cell(nmod, 1);
    %global
    U_invS_g = cell(nmod, 1);
    U_invSMean_g = cell(nmod, 1);
    U_invSMean_prior = cell(nmod, 1);
    U_invS_not = cell(nmod,1);
    U_invSMean_not = cell(nmod,1);
    infty = 1e6;
    for k=1:nmod
        U_invS{k} = repmat(eye(dim)*1/infty, [1, 1, N]);
        U_invSMean{k} = zeros(dim, 1, N);
        U_invS_not{k} = zeros(dim, dim, N);
        U_invSMean_not{k} = zeros(dim, 1, N);
        %start with prior, must be randomly init
        U_invS_g{k} = repmat(eye(dim), [1, 1, nvec(k)]);
        U_invSMean_prior{k} = rand(dim, 1, nvec(k));
        U_invSMean_g{k} =  U_invSMean_prior{k};
    end
    U_cov_not = cell(nmod,1);
    U_mean_not = cell(nmod, 1);
    U_invS_new = cell(nmod, 1);
    U_invSMean_new = cell(nmod, 1);
    %get the global
    for k=1:nmod
        for j=1:length(uind{k})
            U_invS_g{k}(:,:,uind{k}(j)) = sum(U_invS{k}(:,:,data_ind{k}{j}), 3) + eye(dim);
            U_invSMean_g{k}(:, :, uind{k}(j)) = sum(U_invSMean{k}(:, :, data_ind{k}{j}), 3) + U_invSMean_prior{k}(:, :, uind{k}(j));        
        end
    end

    for iter = 1:cfg.max_iter
        u_old = U_invSMean_g;
        for k=1:nmod
            %calibrating 
            U_invS_not{k} = U_invS_g{k}(:, :, train(:,k)) - U_invS{k};
            U_invSMean_not{k} = U_invSMean_g{k}(:, :, train(:,k)) - U_invSMean{k};
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
            %only need to compute once
            s_mean_not = y.*reshape(mtimesx(U_mean_not{k},'t', z_nk),[N,1]); 
            s_var_not = sum(reshape(z_nk2.*U_cov_not{k}, ...
                [dim*dim, N]),1)';
            [s_mean_new, s_var_new] = probit_normal_moments(s_mean_not, s_var_not);
            %increment
            s_i_inv = 1./s_var_new - 1./s_var_not;
            s_i_inv_m = s_mean_new./s_var_new - s_mean_not./s_var_not;
            %fprintf('negative %d\n',sum(s_i_inv<0));
            
            U_invS_new{k} = repmat(reshape(s_i_inv, [1, 1, N]), [dim, dim]).*z_nk2;
            U_invSMean_new{k} = repmat(reshape(y.*s_i_inv_m, [1,1,N]), [dim,1]).*z_nk;
            
            %filter out badcase
            U_invS_new{k}(:,:,s_var_new<0) = U_invS{k}(:,:,s_var_new<0);
            U_invSMean_new{k}(:, :, s_var_new<0) = U_invSMean{k}(:,:,s_var_new<0);
        end
        %damping & update
        diff = 0;
        for k=1:nmod
            U_invS{k} = U_invS{k}*(1-cfg.rho) + U_invS_new{k} *cfg.rho;
            U_invSMean{k} = U_invSMean{k}*(1-cfg.rho) + U_invSMean_new{k} *cfg.rho;
            for j=1:length(uind{k})
                U_invS_g{k}(:,:,uind{k}(j)) = sum(U_invS{k}(:,:,data_ind{k}{j}), 3) + eye(dim);
                U_invSMean_g{k}(:, :, uind{k}(j)) = sum(U_invSMean{k}(:, :, data_ind{k}{j}), 3) + U_invSMean_prior{k}(:, :, uind{k}(j));        
            end
            diff = diff + sum(abs(vec(U_invSMean_g{k}) - vec(u_old{k})));
            %diff = diff + norm(vec(U_invSMean_g{k}) - vec(u_old{k}))/norm(vec(u_old{k}));
        end
        diff = diff/nmod;
        fprintf('iter = %d, diff = %g\n', iter, diff);
        %if mod(iter, 10)==0
        if 1
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
            fprintf('iter = %d, diff = %g, auc = %g\n', iter, diff, auc);
        end
        if diff < cfg.tol
            break;
        end
    end
    


end

