addpath(genpath('../..'));
rng(0);

load('simulation2.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.1;
cfg.max_iter = 2000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= prcep(train,test, ts_mean, ts_var, cfg);
cep.ll = logls;
cep.kl = KLs;
cep.auc = aucs;
cep.time = time;
save('./cep-sim2.mat', 'cep');





