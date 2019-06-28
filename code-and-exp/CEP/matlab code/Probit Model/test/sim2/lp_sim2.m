addpath(genpath('../..'));
rng(0);

load('simulation2.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.1;
cfg.max_iter = 2000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= prlp(train,test, ts_mean, ts_var, cfg);
lp.ll = logls;
lp.kl = KLs;
lp.auc = aucs;
lp.time = time;
save('./lp-sim2.mat', 'lp');





