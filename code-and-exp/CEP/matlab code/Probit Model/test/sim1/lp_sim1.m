addpath(genpath('../..'));
rng(0);

load('simulation1.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.1;
cfg.max_iter = 1000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= prlp(train,test, ts_mean, ts_var, cfg);
lp.ll = logls;
lp.kl = KLs;
lp.auc = aucs;
lp.time = time;
save('./lp-sim1.mat', 'lp');





