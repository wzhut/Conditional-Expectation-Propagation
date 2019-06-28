%CEP for binary tensor
function [U_invS_g, U_invSMean_g] = b_cpcep_zhe_order2( train, test, dim, nvec, cfg)
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
            for i = 1 : N
                [~, p] = chol(cov_new(:,:,i));
                if p > 0
                    cov_new(:,:,i) = inv(1/infty*eye(dim) + U_invS_not{k}(:,:,i));
                end
            end
            
            %second order term for mean
            z_nk = mtimesx_scale(y, z_nk);
            r = 1./sqrt(1+s);
            Sigmaax = mtimesx(U_cov_not{k}, z_nk);
            dr = mtimesx_scale(-r.^3, Sigmaax);
            dr2 = mtimesx_scale(-r.^3, U_cov_not{k}) + mtimesx_scale(3*r.^5, mtimesx(Sigmaax, Sigmaax, 't'));
            dz = mtimesx_scale(r, U_mean_not{k}) + mtimesx_scale(mu, dr);
            dz2 = mtimesx(U_mean_not{k}, dr, 't') + mtimesx(dr, U_mean_not{k}, 't') ...
                + mtimesx_scale(mu, dr2);
            Nz = normpdf(z);
            dN_phi = mtimesx_scale(N_phi.*z.*(N_phi.*Nz - 1), dz);
            dz_dz = mtimesx(dz, dz, 't') + mtimesx_scale(z, dz2);
            dNz_z_dz = mtimesx_scale(Nz.*z, dz2) + mtimesx(dz, mtimesx_scale(Nz.*(1-z.^2),dz), 't');
            dN_phi2 = mtimesx_scale(-z, mtimesx(dz,dN_phi,'t')) - mtimesx_scale(N_phi,dz_dz) ...
                + mtimesx_scale(2*N_phi.*Nz.*z, mtimesx(dz, dN_phi, 't')) ...
                + mtimesx_scale(N_phi.^2, dNz_z_dz);
            
            dN_phi_r = mtimesx_scale(r, dN_phi) + mtimesx_scale(N_phi,dr);
            dN_phi_r2 = mtimesx_scale(r, dN_phi2) + mtimesx(dN_phi, dr, 't') ...
                + mtimesx(dr, dN_phi,'t') + mtimesx_scale(N_phi, dr2);
            tmp = sum(reshape(mtimesx(z_nk2, dN_phi_r2), [dim*dim, N]),1)';
            second_order = 2*mtimesx(mtimesx(z_nk2, U_cov_not{k}).*repmat(dN_phi_r, [1, dim]), 't', ones(dim,1, N)) ...
                + mtimesx_scale(tmp,  mtimesx(U_cov_not{k}, z_nk));
            mean_new = mean_new + 0.5*second_order;
            %get the new sufficient statistics
            U_invS_new{k} = multinv(cov_new);
            U_invSMean_new{k} = mtimesx(U_invS_new{k}, mean_new);
            
        end
        %damping & update
        diff = 0;
        for k=1:nmod
            U_invS{k} = U_invS{k}*(1-cfg.rho) + (U_invS_new{k} - U_invS_not{k}) *cfg.rho;
            
            U_invSMean{k} = U_invSMean{k}*(1-cfg.rho) + (U_invSMean_new{k} - U_invSMean_not{k}) *cfg.rho;
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

