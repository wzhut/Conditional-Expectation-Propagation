function [U_mean_g, U_cov_g, tau] = c_cpcep_zhe( train, test, dim, cfg)
    [N, nmod] = size(train);
    nmod = nmod - 1;
    nvec = max(train(:,1:end-1));
    y = train(:,end);

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
    %about tau
    a0 = 1e-3; 
    b0 = 1e-3;
    a = 1e-3*ones(N,1); 
    b = 1e-3*ones(N,1);
    a_g = a0 + sum(a);
    b_g = b0 + sum(b);
    infty = 1e6;
    for k=1:nmod
        U_invS{k} = repmat(eye(dim)*1/infty, [1, 1, N]);
        U_invSMean{k} = zeros(dim, 1, N);
        U_invS_not{k} = zeros(dim, dim, N);
        U_invSMean_not{k} = zeros(dim, 1, N);
        %start with prior, must be randomly init
        U_invS_g{k} = repmat(eye(dim), [1, 1, nvec(k)]);
        U_invSMean_prior{k} = randn(dim, 1, nvec(k));
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
        a_not = a_g - a;
        b_not = b_g - b;
        tau = a_not./b_not;
        %tau = ones(N,1);
        a_g_new = a_not + 0.5;
        b_g_new = b_not + 0.5*y.^2;
        for k=1:nmod
            z_nk = ones(dim, 1, N);
            z_nk2 = ones(dim, dim, N);
            other_modes = setdiff(1:nmod,k);
            for j=1:length(other_modes)
                mode = other_modes(j);
                z_nk = z_nk.*U_mean_not{mode};
                z_nk2 = z_nk2.*(U_cov_not{mode} + mtimesx(U_mean_not{mode},U_mean_not{mode}, 't'));
            end
            tau_now = repmat(reshape(tau, [1,1,N]), [dim, dim]);        
            U_invS_new{k} = U_invS_not{k} + tau_now.*z_nk2;
            U_invSMean_new{k} = repmat(reshape(y.*tau,[1,1,N]), [dim,1]).*z_nk + U_invSMean_not{k};
            if k==1
                b_g_new = b_g_new - y.*reshape(mtimesx(z_nk, 't', U_mean_not{k}), [N,1]) ...
                    + 0.5 * sum(reshape(z_nk2.*(U_cov_not{k} + mtimesx(U_mean_not{k}, U_mean_not{k}, 't')), [dim*dim, N]) ...
                    ,1)';
            end
        end
        %damping & update
        a = a*(1-cfg.rho) + (a_g_new - a_not)*cfg.rho;
        b = b*(1-cfg.rho) + (b_g_new - b_not)*cfg.rho;
        a_g = a0 + sum(a);
        b_g = b0 + sum(b);
        diff = 0;
        for k=1:nmod
            U_invS{k} = U_invS{k}*(1-cfg.rho) + (U_invS_new{k} - U_invS_not{k})*cfg.rho;
            U_invSMean{k} = U_invSMean{k}*(1-cfg.rho) + (U_invSMean_new{k} - U_invSMean_not{k})*cfg.rho;
            for j=1:length(uind{k})
                U_invS_g{k}(:,:,uind{k}(j)) = sum(U_invS{k}(:,:,data_ind{k}{j}), 3) + eye(dim);
                U_invSMean_g{k}(:, :, uind{k}(j)) = sum(U_invSMean{k}(:, :, data_ind{k}{j}), 3) + U_invSMean_prior{k}(:, :, uind{k}(j));        
            end
            diff = diff + sum(abs(vec(U_invSMean_g{k}) - vec(u_old{k})));
            %diff = diff + norm(vec(U_invSMean_g{k}) - vec(u_old{k}))/norm(vec(u_old{k}));
        end
        diff = diff/nmod;
        %fprintf('iter = %d, diff = %g\n', iter, diff);
        %if mod(iter, 10)==0
        if 1
            pred = ones(dim, 1, size(test,1));
            U_cov_g = cell(nmod,1);
            U_mean_g = cell(nmod,1);
            for k=1:nmod
                U_cov_g{k} = multinv(U_invS_g{k});
                U_mean_g{k} = mtimesx(U_cov_g{k}, U_invSMean_g{k});                
                pred = pred.*U_mean_g{k}(:,:,test(:,k));
            end
            pred = sum(reshape(pred,[dim,size(test,1)]),1)';
            rmse = sqrt(mean((pred - test(:,end)).^2));
            fprintf('iter = %d, diff = %g, tau = %g, rmse = %g\n', iter, diff, a_g/b_g, rmse);
        end
        if diff < cfg.tol
            break;
        end
    end
    


end

