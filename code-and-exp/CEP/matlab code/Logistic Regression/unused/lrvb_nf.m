%Phi: n by p input matrix, each row is a data point
%t: labels, n by 1 vector, in {1, -1}
function [logl, KL, auc, logls, KLs, aucs, time]= lrvb_nf(train, test,ts_mean, ts_var, opt)
% each row of x is a data point
% For classification, t should be {1,-1}.

    % record
    logls = struct('mean', {}, 'std', {});
    KLs = [];
    aucs = [];
    time = [];
    
    % intermediate record
    intermediate_record = struct('joint_var', {}, 'joint_mean', {}, 'time', {});
    
    % gaussian-hermite quadrature
    num_nd = 9;
    [nd,weight] = quadrl(num_nd);
    train_nd = repmat(nd, [size(train.y,1), 1]);
    train_yn = repmat(train.y, [1, num_nd]);

    test_nd = repmat(nd, [size(test.y, 1), 1]);
    test_yn = repmat(test.y, [1, num_nd]);
    % sigma function
    sigma = @(x) 1 ./ (1 + exp(-x));
    
    if nargin < 5
      opt = [];
      opt.max_iter = 500;
      opt.tol = 0; %1e-5;
      opt.verbose = 1;
    end
    
    Phi = train.x;
    t = 2 * train.y - 1;
    
    [n, dim] = size(Phi);
    var_w = eye(dim);
    mean_w = zeros(dim,1);
    wwmtp = var_w +  mean_w*mean_w';
    %s1 = 1e-3; s2 = 1;
    %z = init_z*ones(dim,1);       
    %g0 = ones(dim,1);
    %h0 = 1*ones(dim,1);
    %g = g0;
    %h = h0;
    niter = 0;
    %varational parameter for logistic, epsilon
    %lq = zeros(1,n);
    %for i=1:n
    %	lq(1,i) = sqrt( Phi(i,:)* wwmtp *Phi(i,:)');
    %end
    lq = sqrt(sum((Phi*wwmtp).*Phi,2));
    tic;
    while (niter<opt.max_iter)
        niter = niter + 1;
        A = eye(dim);         
        %%Update varational parater for logistic, lambda(epsilon): lqbnd,
        % lq: episolon
        lqbnd = (1 ./ (4*lq)) .* tanh(lq/2);
        B = 2 * diag( lqbnd );            
        %%update Q(w) Sigma_w
        inv_var_w = A + Phi' * B * Phi;
        var_w =  inv_var_w\eye(size(inv_var_w));
        mean_w_old = mean_w;
        %%update Q(w) mu_w
        mean_w = inv_var_w\(0.5*Phi' * t);
        %%update statistics of w
        wwmtp = var_w +  mean_w*mean_w';
        %%update Q(z)        
        %for i=1:dim
        %    z(i) = 1/(1 + exp(psi(h(i))-psi(g(i)) + 0.5*log(s1/s2) + 0.5*wwmtp(i,i)*(1/s1-1/s2)));
        %end
        %%update varational paramter for logistic, epislon----lq
        %lq = zeros(1,n);
        %for i=1:n
        %    lq(1,i) = sqrt( Phi(i,:)* wwmtp *Phi(i,:)');
        %end  
        lq = sqrt(sum((Phi*wwmtp).*Phi,2));
        time(niter) = toc;
        diff_sum = sum(abs(mean_w_old - mean_w));
        
        % KL
        KL = 0.5 * (logdet(var_w) -  logdet(ts_var) + trace(var_w \ ts_var) + (mean_w - ts_mean)' * inv(var_w) * (mean_w - ts_mean) - dim);
        % loglikelihood
        % train
        proj_mean = train.x * mean_w;
        proj_var = diag(train.x * var_w * train.x');
        theta_s = train_nd .* (repmat(sqrt(proj_var), [1, num_nd])) + repmat(proj_mean, [1, num_nd]);
        sigma_tx = sigma(theta_s);
        h = sigma_tx;
        idx0 = find(train_yn == 0);
        h(idx0) = 1 - h(idx0);
        train_logl = mean(log(h * weight + realmin));
        % test
        proj_mean = test.x * mean_w;
        proj_var = diag(test.x * var_w * test.x');
        theta_s = test_nd .* (repmat(sqrt(proj_var), [1, num_nd])) + repmat(proj_mean, [1, num_nd]);
        sigma_tx = sigma(theta_s);
        h = sigma_tx;
        idx0 = find(test_yn == 0);
        h(idx0) = 1 - h(idx0);
        tmp = log(h * weight + realmin);
        logl.mean = mean(tmp);
        logl.std = std(tmp);

        p = sigma(test.x * mean_w);
        [~,~,~,auc] = perfcurve(test.y,p,1);

        % record
        KLs(end+1) = KL;
        logls(end+1) = logl;
        aucs(end+1) = auc;
        
        r.joint_var = var_w;
        r.joint_mean = mean_w;
        r.time = time(niter);
        intermediate_record(end+1) = r;
        
        if opt.verbose
%             fprintf(2,'niter= %d, diff_sum=%f\n',niter, diff_sum);      
             disp(sprintf('lrvb_nf -- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',niter, diff_sum, train_logl, logl.mean ,auc, KL));
        end
        if  ( diff_sum < opt.tol )
            save('./intermediate/lrlp_nf.mat', 'intermediate_record');
            break; % stop VB iteration
        end    
        
        
    end
end





