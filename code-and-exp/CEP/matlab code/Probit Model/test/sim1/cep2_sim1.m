addpath(genpath('../..'));
rng(0);

load('simulation1.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.1;
cfg.cep_iter = 0;
cfg.max_iter = 1000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= prcep2_diag(train,test, ts_mean, ts_var, cfg);
cep2 ={};
cep2.ll = logls;
cep2.kl = KLs;
cep2.auc = aucs;
cep2.time = time;
save('./cep2-sim1.mat', 'cep2');





