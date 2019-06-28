addpath(genpath('../..'));
rng(0);

load('simulation1.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.01;
cfg.max_iter = 1000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= prlp_nf(train,test, ts_mean, ts_var, cfg);
lpnf.ll = logls;
lpnf.kl = KLs;
lpnf.auc = aucs;
lpnf.time = time;
save('./lpnf-sim1.mat', 'lpnf');





