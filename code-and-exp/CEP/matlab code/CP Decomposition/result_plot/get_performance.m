clear;

num_rank = 4;
num_fold = 5;

auc = zeros(num_rank, num_fold);
rmse = zeros(num_rank, num_fold);

%% binary

% CEP
load('cep-binary-performance.mat');
for r = 1 : num_rank
    for nfold = 1 : num_fold
        iter = cep.iter(r,nfold);
        auc(r, nfold) = cep.test_auc_stat{r, nfold}(iter); 
    end
end
cep_auc = auc;

% VI
load('vi-binary-performance.mat');
for r = 1 : num_rank
    for nfold = 1 : num_fold
        iter = vi.iter(r,nfold);
        auc(r, nfold) = vi.test_auc_stat{r, nfold}(iter); 
    end
end
vi_auc = auc;

% LP
load('lp-binary-performance.mat');
for r = 1 : num_rank
    for nfold = 1 : num_fold
        iter = lp.iter(r,nfold);
        auc(r, nfold) = lp.test_auc_stat{r, nfold}(iter); 
    end
end
lp_auc = auc;




%% continuous
% CEP
load('cep-continuous-performance.mat');
for r = 1 : num_rank
    for nfold = 1 : num_fold
        iter = cep.iter(r,nfold);
        rmse(r, nfold) = cep.rmse_stat{r, nfold}(iter); 
    end
end
cep_rmse = rmse;

% VI
load('vi-continuous-performance.mat');
for r = 1 : num_rank
    for nfold = 1 : num_fold
        iter = vi.iter(r,nfold);
        rmse(r, nfold) = vi.rmse_stat{r, nfold}(iter); 
    end
end
vi_rmse = rmse;

% LP
load('lp-continuous-performance.mat');
for r = 1 : num_rank
    for nfold = 1 : num_fold
        iter = lp.iter(r,nfold);
        rmse(r, nfold) = lp.rmse_stat{r, nfold}(iter); 
    end
end
lp_rmse = rmse;