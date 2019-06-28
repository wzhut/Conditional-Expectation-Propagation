%Phi: n*p matrix, each row is a data point
%t should be in {-1,1}.
%notice that: init_z tends to be very important and the selection result is
%very sensitive to the configuration of init_z. Generally, it should be
%bigger than 0.9; Otherwise none of the variables can be selected.
function [logl, KL, auc, logls, KLs, aucs, time] = prvb(train, test,ts_mean, ts_var, opt)

    % record
    logls = [];
    KLs = [];
    aucs = [];
    time = [];
    
    % intermediate record
    intermediate_record = struct('joint_var', {}, 'joint_mean', {}, 'time', {});

    
    if nargin < 5
      opt = [];
      opt.max_iter = 1000;
      opt.tol = 0;
      opt.verbose = 1;
    end
    Phi = train.x;
    t = 2 * train.y - 1;
    
    [~, dim] = size(Phi);
    mean_w = zeros(dim,1);
    var_w = ones(dim,1);
    y = t;
    %t = (t<0).*ones(n,1) + (t>0).*ones(n,1)*2;
    niter = 0;
    %b = [-inf;0;inf];
    tic;
    while (niter<opt.max_iter)
        niter = niter + 1;
        mean_w_old = mean_w;
        for k=1:dim
            var_w(k) = 1/(1 + sum(Phi(:,k).^2));
            ind = setdiff(1:dim, k);
            mean_w(k) = var_w(k)*(y'*Phi(:,k)  ...
                -sum(sum(Phi(:,ind)*diag(mean_w(ind)),2).*Phi(:,k)));
        end
        time(end+1) = toc;
%         A = eye(dim);
%         inv_var_w = A + Phi'*Phi;
%         mean_w_old = mean_w;
%         %%update Q(w) mu_w
%         mean_w = inv_var_w\(Phi' * y);
        diff_sum = sum(abs(mean_w_old - mean_w));
        
        % KL divergence
   
        KL = 0.5 * (logdet(diag(var_w)) -  logdet(ts_var) + trace(diag(var_w) \ ts_var) + (mean_w - ts_mean)' / diag(var_w) * (mean_w - ts_mean) - dim);

        % log likelihood

        t1 = (2* train.y - 1) .* (train.x * mean_w);
        t2 = sqrt(1 + train.x.^2 * var_w);
        train_logl = mean(normcdfln(t1 ./ t2));

        t1 = (2* test.y - 1) .* (test.x * mean_w);
        t2 = sqrt(1 + test.x.^2 * var_w);

        p = normcdf(test.x * mean_w);
        [~,~,~,auc] = perfcurve(test.y,p,1);

        tmp = normcdfln(t1 ./ t2);
        logl = mean(tmp);

        % record
        KLs(end+1) = KL;
        logls(end+1) = logl;
        aucs(end+1) = auc; 
        
        r.joint_var = diag(var_w);
        r.joint_mean = mean_w;
        r.time = time(niter);
        intermediate_record(end+1) = r;
        
        if opt.verbose
%             fprintf(2,'niter= %d, diff_sum=%f\n',niter, diff_sum);      
             disp(sprintf('prvb -- iter:%d, diff:%.8f, train_logl:%.8f, test_logl:%.8f, auc:%.8f,  KL:%.8f',niter, diff_sum, train_logl, logl ,auc, KL));
        end
        if  (diff_sum < opt.tol )
            save('./intermediate/prvb.mat', 'intermediate_record');
            break; % stop VB iteration
        end
        %%update Q(y)--probit regression
        my = Phi*mean_w;
        %y = my + (normpdf(b(t),my,ones(n,1))-normpdf(b(t+1),my,ones(n,1)))./(normcdf(b(t+1)-my) - normcdf(b(t)-my));
        y = my + t.*normpdf(my)./normcdf(t.*my);
    end
end





