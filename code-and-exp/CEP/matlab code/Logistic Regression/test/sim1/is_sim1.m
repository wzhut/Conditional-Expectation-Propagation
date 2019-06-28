addpath(genpath('../..'));
rng(0);

load('simulation1.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.1;
cfg.cep_iter = 0;
cfg.max_iter = 1000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= lrepis(train,test, ts_mean, ts_var);
is ={};
is.ll = logls;
is.kl = KLs;
is.auc = aucs;
is.time = time;
save('./is-sim1.mat', 'is');





