%opt.s1 >> opt.s2, e.g., opt.s1 = 1, opt.s2 = 1e-6
%opt.task: 0: regression, 1: classification 
%for classification, we are using logistic regression
%for regression, we are using Gaussian likelihood
%output: w: mean_w; v: mean_v; mean_p: mean prob; var_p: variance prob
function [w, v, mean_p, var_p] = large_scale_ss(X, y, opt)
    %%step 1, MAP estimate p(y,w|X)
      %w_init = (X'*X+ 1e-0*eye(size(X,2)))\(X'*y); % this is for use of test posterior for classification
      %w_init = (X'*X)\(X'*y);
      %w_init = zeros(size(X,2),1);
      w_init = opt.w_init;
      %w_init = 0*randn(size(X,2),1);%ones(size(X,2),1);
      
      other_args = {X, y, opt};      
      minf_opt = [];
      minf_opt.display = 'none';
      %check gradient
      %fastDerivativeCheck(@func_grad, w_init, 1, 2, other_args{:});%central finite difference method
      [out] = minFunc(@func_grad, w_init , minf_opt, other_args{:});
      w = out;          
    %step 2, Laplace approx. p(w|X,y)
    [~,~,h] = func_grad(w, X, y, opt);
    v = 1./h;
    %step 3, recover p(z|X,y) by using quadrature
    dim = size(X,2);
    mean_p = zeros(dim,1);
    var_p = zeros(dim,1);
    
    for i=1:dim                 
        %%using quadrature
        [nd,weights] = quadrl(9);
        x = w(i) + sqrt(v(i))*nd;
        mean_p(i) = func_quad_v2(x,2/3,1/3,opt.s1,opt.s2)*weights;
        var_p(i) = func_quad_v2(x,1/2,1/6,opt.s1,opt.s2)*weights;
    end
    var_p = var_p - mean_p.^2;
    var_p(var_p<0) = 0;
end