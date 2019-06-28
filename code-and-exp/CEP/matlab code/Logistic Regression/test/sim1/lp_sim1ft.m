addpath(genpath('../..'));
rng(0);

load('simulation1.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.05;
cfg.max_iter = 0;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= lrlp_first_iter(train,test, ts_mean, ts_var, cfg);
lpft.ll = logls;
lpft.kl = KLs;
lpft.auc = aucs;
lpft.time = time;
save('./lp-sim1-first.mat', 'lpft');





