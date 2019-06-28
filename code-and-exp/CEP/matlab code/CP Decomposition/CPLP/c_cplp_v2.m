function [mse, rmse, iter, diffs, test_rmses, time] = c_cplp_v2(train, test, dim, cfg)
    [N, nmod] = size(train);
    nmod = nmod - 1;
    nvec = max(train(:,1:end-1));
    y = train(:,end);
    max_iter = cfg.max_iter;
    train_rmses = zeros([max_iter, 1]);
    test_rmses = zeros([max_iter, 1]);
    time = zeros([max_iter, 1]);

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
        U_invSMean_prior{k} = rand(dim, 1, nvec(k));
    end
    U_cov_not = cell(nmod,1);
    U_mean_not = cell(nmod, 1);
    U_invS_new_inc = cell(nmod, 1);
    U_invSMean_new_inc = cell(nmod, 1);
    %get the global
    for k=1:nmod
        for j=1:length(uind{k})
            U_invS_g{k}(:,:,uind{k}(j)) = sum(U_invS{k}(:,:,data_ind{k}{j}), 3) + eye(dim);
            U_invSMean_g{k}(:, :, uind{k}(j)) = sum(U_invSMean{k}(:, :, data_ind{k}{j}), 3) + U_invSMean_prior{k}(:, :, uind{k}(j));        
        end
        U_invS_new_inc{k} = zeros(dim, dim, N);
        U_invSMean_new_inc{k} = zeros(dim, 1, N);
    end
    tic;
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
        a_g_new_inc = zeros(N, 1);
        b_g_new_inc = zeros(N, 1);
        
        
        for i = 1 : N
            param.y = y(i);
            param.r = dim;
            param.mu = cell(nmod, 1);
            param.precision = cell(nmod, 1);
            param.alpha = a_not(i);
            param.beta = b_not(i);
            x0 = rand(dim * nmod + 1, 1);
            x0(end) = log(a_not(i) / b_not(i));
            for k = 1 : nmod
                param.mu{k} = U_mean_not{k}(:,:,train(i, k));
                param.precision{k} = U_invS_not{k}(:,:,train(i,k));%q_precision(:,:,m,i);
                x0((k - 1) * dim + 1: k * dim) = U_mean_not{k}(:,:,train(i, k));%q_mean(:,:,m,i);
            end
            options = [];
            options.display = 'none';
            options.Method = 'lbfgs';
%             options.DERIVATIVECHECK = 'on';
            func = @(x) c_fun(x, param);
            [opt_x] = minFunc(func, x0, options);
            
             for k = 1 : nmod
                % hessian
                other_modes = setdiff(1:nmod, k);
                tmp = ones(dim,1);
                for j = 1 : nmod - 1
                    tmp = tmp .* opt_x((other_modes(j) - 1) * dim + 1: other_modes(j) * dim);
                end

                mean_value = opt_x((k - 1) * dim + 1 : k * dim);
                U_invS_new_inc{k}(:,:,train(i,k)) = exp(opt_x(end)) * (tmp * tmp');

                if U_invS_new_inc{k}(:,:,train(i,k)) < dim
                  U_invS_new_inc{k}(:,:,train(i,k)) = eye(dim) * 1e-6;
                end
                precision = U_invS_new_inc{k}(:,:,train(i,k)) + U_invS_not{k}(:,:,train(i,k));
                U_invSMean_new_inc{k}(:,:,train(i,k)) = precision * mean_value - U_invSMean_not{k}(:,:,train(i,k));
             end
            % log tau
            mean_lt = opt_x(end);

            for k = 1 : nmod
                tmp = ones(dim,1);
                tmp = tmp .* opt_x((k - 1) * dim + 1: k * dim);
            end
            var_lt = 0.5 * (sum(tmp) - y(i)) ^ 2 + exp(mean_lt) * b_not(i);
            mean_t = exp(mean_lt + var_lt / 2);
            var_t = exp(2 * mean_lt + var_lt) * (exp(var_lt) - 1) + realmin;


            joint_beta = mean_t / var_t;
            joint_alpha = joint_beta * mean_t;
            a_g_new_inc(i) = joint_alpha - a_not(i) + 1;
            if a_g_new_inc(i) < 1
                a_g_new_inc(i) = 1;
            end
            b_g_new_inc(i) = joint_beta - b_not(i);
            if b_g_new_inc(i) < 0
                b_g_new_inc(i) = 0.01;
            end
        end
        
        %damping & update
        a = a*(1-cfg.rho) + a_g_new_inc*cfg.rho;
        b = b*(1-cfg.rho) + b_g_new_inc*cfg.rho;
        a_g = a0 + sum(a);
        b_g = b0 + sum(b);
        diff = 0;
        for k=1:nmod
            U_invS{k} = U_invS{k} * (1-cfg.rho) + U_invS_new_inc{k} * cfg.rho;
            U_invSMean{k} = U_invSMean{k}*(1-cfg.rho) + U_invSMean_new_inc{k}*cfg.rho;
            for j=1:length(uind{k})
                U_invS_g{k}(:,:,uind{k}(j)) = sum(U_invS{k}(:,:,data_ind{k}{j}), 3) + eye(dim);
                U_invSMean_g{k}(:, :, uind{k}(j)) = sum(U_invSMean{k}(:, :, data_ind{k}{j}), 3) + U_invSMean_prior{k}(:, :, uind{k}(j));        
            end
            diff = diff + sum(abs(vec(U_invSMean_g{k}) - vec(u_old{k})));
        end
        diff = diff/nmod;
        time(iter) = toc;
        
        % train
        pred = ones(dim, 1, size(train,1));
        U_cov_g = cell(nmod,1);
        U_mean_g = cell(nmod,1);
        for k=1:nmod
            U_cov_g{k} = multinv(U_invS_g{k});
            U_mean_g{k} = mtimesx(U_cov_g{k}, U_invSMean_g{k});                
            pred = pred.*U_mean_g{k}(:,:,train(:,k));
        end
        pred = sum(reshape(pred,[dim,size(train,1)]),1)';
        rmse = sqrt(mean((pred - train(:,end)).^2));
        train_rmses(iter) = rmse;
        
        % test
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
        test_rmses(iter) = rmse;
        
        if cfg.verbose == 1
            fprintf('iter = %d, diff = %g, tau = %g, train_rmse = %g, test_rmse = %g\n', iter, diff, a_g/b_g, train_rmses(iter), test_rmses(iter));
        end
        if diff < cfg.tol
            break;
        end        
        if iter > 10 && mean(abs(train_rmses(iter-9: iter-1) - train_rmses(iter))) < cfg.tol
            break;
        end
    end
end